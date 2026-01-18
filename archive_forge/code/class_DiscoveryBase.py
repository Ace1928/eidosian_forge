from keystoneauth1 import _utils as utils
class DiscoveryBase(dict):
    """The basic version discovery structure.

    All version discovery elements should have access to these values.

    :param string id: The version id for this version entry.
    :param string status: The status of this entry.
    :param DateTime updated: When the API was last updated.
    """

    def __init__(self, id, status=None, updated=None):
        super(DiscoveryBase, self).__init__()
        self.id = id
        self.status = status or 'stable'
        self.updated = updated or utils.before_utcnow(days=_DEFAULT_DAYS_AGO)

    @property
    def id(self):
        return self.get('id')

    @id.setter
    def id(self, value):
        self['id'] = value

    @property
    def status(self):
        return self.get('status')

    @status.setter
    def status(self, value):
        self['status'] = value

    @property
    def links(self):
        return self.setdefault('links', [])

    @property
    def updated_str(self):
        return self.get('updated')

    @updated_str.setter
    def updated_str(self, value):
        self['updated'] = value

    @property
    def updated(self):
        return utils.parse_isotime(self.updated_str)

    @updated.setter
    def updated(self, value):
        self.updated_str = value.isoformat()

    def add_link(self, href, rel='self', type=None):
        link = {'href': href, 'rel': rel}
        if type:
            link['type'] = type
        self.links.append(link)
        return link

    @property
    def media_types(self):
        return self.setdefault('media-types', [])

    def add_media_type(self, base, type):
        mt = {'base': base, 'type': type}
        self.media_types.append(mt)
        return mt