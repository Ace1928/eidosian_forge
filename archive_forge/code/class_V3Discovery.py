from keystoneauth1 import _utils as utils
class V3Discovery(DiscoveryBase):
    """A Version element for a V3 identity service endpoint.

    Provides some default values and helper methods for creating a v3
    endpoint version structure. Clients should use this instead of creating
    their own structures.

    :param href: The url that this entry should point to.
    :param string id: The version id that should be reported. (optional)
                      Defaults to 'v3.0'.
    :param bool json: Add JSON media-type elements to the structure.
    :param bool xml: Add XML media-type elements to the structure.
    """

    def __init__(self, href, id=None, json=True, xml=True, **kwargs):
        super(V3Discovery, self).__init__(id or 'v3.0', **kwargs)
        self.add_link(href)
        if json:
            self.add_json_media_type()
        if xml:
            self.add_xml_media_type()

    def add_json_media_type(self):
        """Add the JSON media-type links.

        The standard structure includes a list of media-types that the endpoint
        supports. Add JSON to the list.
        """
        self.add_media_type(base='application/json', type='application/vnd.openstack.identity-v3+json')

    def add_xml_media_type(self):
        """Add the XML media-type links.

        The standard structure includes a list of media-types that the endpoint
        supports. Add XML to the list.
        """
        self.add_media_type(base='application/xml', type='application/vnd.openstack.identity-v3+xml')