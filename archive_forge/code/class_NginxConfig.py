from ._base import *
class NginxConfig(object):
    """
    Represents an nginx configuration.
    A `NginxConfig` can consist of any number of server blocks, as well as Upstream
    and other types of containers. It can also include top-level comments.
    """

    def __init__(self, *args):
        """
        Initialize object.
        :param *args: Any objects to include in this NginxConfig.
        """
        self.children = list(args)

    def add(self, *args):
        """
        Add object(s) to the NginxConfig.
        :param *args: Any objects to add to the NginxConfig.
        :returns: full list of NginxConfig's child objects
        """
        self.children.extend(args)
        return self.children

    def remove(self, *args):
        """
        Remove object(s) from the NginxConfig.
        :param *args: Any objects to remove from the NginxConfig.
        :returns: full list of NginxConfig's child objects
        """
        for x in args:
            self.children.remove(x)
        return self.children

    def filter(self, btype='', name=''):
        """
        Return child object(s) of this NginxConfig that satisfy certain criteria.
        :param str btype: Type of object to filter by (e.g. 'Key')
        :param str name: Name of key OR container value to filter by
        :returns: full list of matching child objects
        """
        filtered = []
        for x in self.children:
            if name and isinstance(x, Key) and (x.name == name):
                filtered.append(x)
            elif isinstance(x, Container) and x.__class__.__name__ == btype and (x.value == name):
                filtered.append(x)
            elif not name and btype and (x.__class__.__name__ == btype):
                filtered.append(x)
        return filtered

    @property
    def servers(self):
        """Return a list of child Server objects."""
        return [x for x in self.children if isinstance(x, Server)]

    @property
    def server(self):
        """Convenience property to fetch the first available server only."""
        return self.servers[0]

    @property
    def as_list(self):
        """Return all child objects in nested lists of strings."""
        return [x.as_list for x in self.children]

    @property
    def as_dict(self):
        """Return all child objects in nested dict."""
        return {'conf': [x.as_dict for x in self.children]}

    @property
    def as_strings(self):
        """Return the entire NginxConfig as nginx config strings."""
        ret = []
        for x in self.children:
            if isinstance(x, (Key, Comment)):
                ret.append(x.as_strings)
            else:
                for y in x.as_strings:
                    ret.append(y)
        if ret:
            ret[-1] = re.sub('}\n+$', '}\n', ret[-1])
        return ret