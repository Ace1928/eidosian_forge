from urllib import parse
from openstack import format
class HREFToUUID(format.Formatter):

    @classmethod
    def deserialize(cls, value):
        """Convert a HREF to the UUID portion"""
        parts = parse.urlsplit(value)
        if not all(parts[:3]):
            raise ValueError('Unable to convert %s to an ID' % value)
        return parts.path.split('/')[-1]