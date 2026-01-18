from __future__ import absolute_import, division, print_function
import os
import re
import traceback
import shutil
import tempfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
class PgHbaRule(dict):
    """
    This class represents one rule as defined in a line in a PgHbaFile.
    """

    def __init__(self, contype=None, databases=None, users=None, source=None, netmask=None, method=None, options=None, line=None, comment=None):
        """
        This function can be called with a comma separated list of databases and a comma separated
        list of users and it will act as a generator that returns a expanded list of rules one by
        one.
        """
        super(PgHbaRule, self).__init__()
        if line:
            self.fromline(line)
        if comment:
            self['comment'] = comment
        rule = dict(zip(PG_HBA_HDR, [contype, databases, users, source, netmask, method, options]))
        for key, value in rule.items():
            if value:
                self[key] = value
        for key in ['method', 'type']:
            if key not in self:
                raise PgHbaRuleError('Missing {method} in rule {rule}'.format(method=key, rule=self))
        if self['method'] not in PG_HBA_METHODS:
            msg = "invalid method {method} (should be one of '{valid_methods}')."
            raise PgHbaRuleValueError(msg.format(method=self['method'], valid_methods="', '".join(PG_HBA_METHODS)))
        if self['type'] not in PG_HBA_TYPES:
            msg = "invalid connection type {0} (should be one of '{1}')."
            raise PgHbaRuleValueError(msg.format(self['type'], "', '".join(PG_HBA_TYPES)))
        if self['type'] == 'local':
            self.unset('src')
            self.unset('mask')
        elif 'src' not in self:
            raise PgHbaRuleError('Missing src in rule {rule}'.format(rule=self))
        elif '/' in self['src']:
            self.unset('mask')
        else:
            self['src'] = str(self.source())
            self.unset('mask')

    def unset(self, key):
        """
        This method is used to unset certain columns if they exist
        """
        if key in self:
            del self[key]

    def line(self):
        """
        This method can be used to return (or generate) the line
        """
        try:
            return self['line']
        except KeyError:
            self['line'] = '\t'.join([self[k] for k in PG_HBA_HDR if k in self.keys()])
            return self['line']

    def fromline(self, line):
        """
        split into 'type', 'db', 'usr', 'src', 'mask', 'method', 'options' cols
        """
        if WHITESPACES_RE.sub('', line) == '':
            return
        cols = WHITESPACES_RE.split(line)
        if len(cols) < 4:
            msg = 'Rule {0} has too few columns.'
            raise PgHbaValueError(msg.format(line))
        if cols[0] not in PG_HBA_TYPES:
            msg = 'Rule {0} has unknown type: {1}.'
            raise PgHbaValueError(msg.format(line, cols[0]))
        if cols[0] == 'local':
            cols.insert(3, None)
            cols.insert(3, None)
        if len(cols) < 6:
            cols.insert(4, None)
        elif cols[5] not in PG_HBA_METHODS:
            cols.insert(4, None)
        if cols[5] not in PG_HBA_METHODS:
            raise PgHbaValueError("Rule {0} of '{1}' type has invalid auth-method '{2}'".format(line, cols[0], cols[5]))
        if len(cols) < 7:
            cols.insert(6, None)
        else:
            cols[6] = ' '.join(cols[6:])
        rule = dict(zip(PG_HBA_HDR, cols[:7]))
        for key, value in rule.items():
            if value:
                self[key] = value

    def key(self):
        """
        This method can be used to get the key from a rule.
        """
        if self['type'] == 'local':
            source = 'local'
        else:
            source = str(self.source())
        return (source, self['db'], self['usr'])

    def source(self):
        """
        This method is used to get the source of a rule as an ipaddress object if possible.
        """
        if 'mask' in self.keys():
            try:
                ipaddress.ip_address(u'{0}'.format(self['src']))
            except ValueError:
                raise PgHbaValueError('Mask was specified, but source "{0}" is not valid ip'.format(self['src']))
            try:
                mask_as_ip = ipaddress.ip_address(u'{0}'.format(self['mask']))
            except ValueError:
                raise PgHbaValueError('Mask {0} seems to be invalid'.format(self['mask']))
            binvalue = '{0:b}'.format(int(mask_as_ip))
            if '01' in binvalue:
                raise PgHbaValueError('IP mask {0} seems invalid (binary value has 1 after 0)'.format(self['mask']))
            prefixlen = binvalue.count('1')
            sourcenw = '{0}/{1}'.format(self['src'], prefixlen)
            try:
                return ipaddress.ip_network(u'{0}'.format(sourcenw), strict=False)
            except ValueError:
                raise PgHbaValueError('{0} is not valid address range'.format(sourcenw))
        try:
            return ipaddress.ip_network(u'{0}'.format(self['src']), strict=False)
        except ValueError:
            return self['src']

    def __lt__(self, other):
        """This function helps sorted to decide how to sort.

        It just checks itself against the other and decides on some key values
        if it should be sorted higher or lower in the list.
        The way it works:
        For networks, every 1 in 'netmask in binary' makes the subnet more specific.
        Therefore I chose to use prefix as the weight.
        So a single IP (/32) should have twice the weight of a /16 network.
        To keep everything in the same weight scale,
        - for ipv6, we use a weight scale of 0 (all possible ipv6 addresses) to 128 (single ip)
        - for ipv4, we use a weight scale of 0 (all possible ipv4 addresses) to 128 (single ip)
        Therefore for ipv4, we use prefixlen (0-32) * 4 for weight,
        which corresponds to ipv6 (0-128).
        """
        myweight = self.source_weight()
        hisweight = other.source_weight()
        if myweight != hisweight:
            return myweight > hisweight
        myweight = self.db_weight()
        hisweight = other.db_weight()
        if myweight != hisweight:
            return myweight < hisweight
        myweight = self.user_weight()
        hisweight = other.user_weight()
        if myweight != hisweight:
            return myweight < hisweight
        try:
            return self['src'] < other['src']
        except TypeError:
            return self.source_type_weight() < other.source_type_weight()
        except Exception:
            return self.line() < other.line()

    def source_weight(self):
        """Report the weight of this source net.

        Basically this is the netmask, where IPv4 is normalized to IPv6
        (IPv4/32 has the same weight as IPv6/128).
        """
        if self['type'] == 'local':
            return 130
        sourceobj = self.source()
        if isinstance(sourceobj, ipaddress.IPv4Network):
            return sourceobj.prefixlen * 4
        if isinstance(sourceobj, ipaddress.IPv6Network):
            return sourceobj.prefixlen
        if isinstance(sourceobj, str):
            if sourceobj == 'all':
                return 0
            if sourceobj == 'samehost':
                return 129
            if sourceobj == 'samenet':
                return 96
            if sourceobj[0] == '.':
                return 64
            return 128
        raise PgHbaValueError('Cannot deduct the source weight of this source {sourceobj}'.format(sourceobj=sourceobj))

    def source_type_weight(self):
        """Give a weight on the type of this source.

        Basically make sure that IPv6Networks are sorted higher than IPv4Networks.
        This is a 'when all else fails' solution in __lt__.
        """
        if self['type'] == 'local':
            return 3
        sourceobj = self.source()
        if isinstance(sourceobj, ipaddress.IPv4Network):
            return 2
        if isinstance(sourceobj, ipaddress.IPv6Network):
            return 1
        if isinstance(sourceobj, str):
            return 0
        raise PgHbaValueError('This source {0} is of an unknown type...'.format(sourceobj))

    def db_weight(self):
        """Report the weight of the database.

        Normally, just 1, but for replication this is 0, and for 'all', this is more than 2.
        """
        if self['db'] == 'all':
            return 100000
        if self['db'] == 'replication':
            return 0
        if self['db'] in ['samerole', 'samegroup']:
            return 1
        return 1 + self['db'].count(',')

    def user_weight(self):
        """Report weight when comparing users."""
        if self['usr'] == 'all':
            return 1000000
        return 1