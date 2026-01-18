from __future__ import absolute_import, division, print_function
import os.path
import re
from ansible_collections.community.dns.plugins.module_utils.names import InvalidDomainName, split_into_labels, normalize_label
class PublicSuffixList(object):
    """
    Contains the Public Suffix List.
    """

    def __init__(self, rules):
        self._generic_rule = PublicSuffixEntry(('*',))
        self._rules = sorted(rules, key=lambda entry: entry.labels)

    @classmethod
    def load(cls, filename):
        """
        Load Public Suffix List from the given filename.
        """
        rules = []
        part = None
        with open(filename, 'rb') as content_file:
            content = content_file.read().decode('utf-8')
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('//') or not line:
                m = _BEGIN_SUBSET_MATCHER.search(line)
                if m:
                    part = m.group(1).lower()
                m = _END_SUBSET_MATCHER.search(line)
                if m:
                    part = None
                continue
            if part is None:
                raise Exception('Internal error: found PSL entry with no part!')
            exception_rule = False
            if line.startswith('!'):
                exception_rule = True
                line = line[1:]
            if line.startswith('.'):
                line = line[1:]
            labels = tuple((normalize_label(label) for label in split_into_labels(line)[0]))
            rules.append(PublicSuffixEntry(labels, exception_rule=exception_rule, part=part))
        return cls(rules)

    def get_suffix_length_and_rule(self, normalized_labels, icann_only=False):
        """
        Given a list of normalized labels, searches for a matching rule.

        Returns the tuple ``(suffix_length, rule)``. The ``rule`` is never ``None``
        except if ``normalized_labels`` is empty, in which case ``(0, None)`` is returned.

        If ``icann_only`` is set to ``True``, only official ICANN rules are used. If
        ``icann_only`` is ``False`` (default), also private rules are used.
        """
        if not normalized_labels:
            return (0, None)
        rules = []
        for rule in self._rules:
            if icann_only and rule.part != 'icann':
                continue
            if rule.matches(normalized_labels):
                rules.append(rule)
        if not rules:
            rules.append(self._generic_rule)
        rule = select_prevailing_rule(rules)
        suffix_length = len(rule.labels)
        if rule.exception_rule:
            suffix_length -= 1
        return (suffix_length, rule)

    def get_suffix(self, domain, keep_unknown_suffix=True, normalize_result=False, icann_only=False):
        """
        Given a domain name, extracts the public suffix.

        If ``keep_unknown_suffix`` is set to ``False``, only suffixes matching explicit
        entries from the PSL are returned. If ``keep_unknown_suffix`` is ``True`` (default),
        the implicit ``*`` rule is used if no other rule matches.

        If ``normalize_result`` is set to ``True``, the result is re-combined form the
        normalized labels. In that case, the result is lower-case ASCII. If
        ``normalize_result`` is ``False`` (default), the result ``result`` always satisfies
        ``domain.endswith(result)``.

        If ``icann_only`` is set to ``True``, only official ICANN rules are used. If
        ``icann_only`` is ``False`` (default), also private rules are used.
        """
        try:
            labels, tail = split_into_labels(domain)
            normalized_labels = [normalize_label(label) for label in labels]
        except InvalidDomainName:
            return ''
        if normalize_result:
            labels = normalized_labels
        suffix_length, rule = self.get_suffix_length_and_rule(normalized_labels, icann_only=icann_only)
        if rule is None:
            return ''
        if not keep_unknown_suffix and rule is self._generic_rule:
            return ''
        return '.'.join(reversed(labels[:suffix_length])) + tail

    def get_registrable_domain(self, domain, keep_unknown_suffix=True, only_if_registerable=True, normalize_result=False, icann_only=False):
        """
        Given a domain name, extracts the registrable domain. This is the public suffix
        including the last label before the suffix.

        If ``keep_unknown_suffix`` is set to ``False``, only suffixes matching explicit
        entries from the PSL are returned. If no suffix can be found, ``''`` is returned.
        If ``keep_unknown_suffix`` is ``True`` (default), the implicit ``*`` rule is used
        if no other rule matches.

        If ``only_if_registerable`` is set to ``False``, the public suffix is returned
        if there is no label before the suffix. If ``only_if_registerable`` is ``True``
        (default), ``''`` is returned in that case.

        If ``normalize_result`` is set to ``True``, the result is re-combined form the
        normalized labels. In that case, the result is lower-case ASCII. If
        ``normalize_result`` is ``False`` (default), the result ``result`` always satisfies
        ``domain.endswith(result)``.

        If ``icann_only`` is set to ``True``, only official ICANN rules are used. If
        ``icann_only`` is ``False`` (default), also private rules are used.
        """
        try:
            labels, tail = split_into_labels(domain)
            normalized_labels = [normalize_label(label) for label in labels]
        except InvalidDomainName:
            return ''
        if normalize_result:
            labels = normalized_labels
        suffix_length, rule = self.get_suffix_length_and_rule(normalized_labels, icann_only=icann_only)
        if rule is None:
            return ''
        if not keep_unknown_suffix and rule is self._generic_rule:
            return ''
        if suffix_length < len(labels):
            suffix_length += 1
        elif only_if_registerable:
            return ''
        return '.'.join(reversed(labels[:suffix_length])) + tail