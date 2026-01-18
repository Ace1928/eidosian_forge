from keystoneauth1 import _utils as utils
class DiscoveryList(dict):
    """A List of version elements.

    Creates a correctly structured list of identity service endpoints for
    use in testing with discovery.

    :param string href: The url that this should be based at.
    :param bool v2: Add a v2 element.
    :param bool v3: Add a v3 element.
    :param string v2_status: The status to use for the v2 element.
    :param DateTime v2_updated: The update time to use for the v2 element.
    :param bool v2_html: True to add a html link to the v2 element.
    :param bool v2_pdf: True to add a pdf link to the v2 element.
    :param string v3_status: The status to use for the v3 element.
    :param DateTime v3_updated: The update time to use for the v3 element.
    :param bool v3_json: True to add a html link to the v2 element.
    :param bool v3_xml: True to add a pdf link to the v2 element.
    """
    TEST_URL = 'http://keystone.host:5000/'

    def __init__(self, href=None, v2=True, v3=True, v2_id=None, v3_id=None, v2_status=None, v2_updated=None, v2_html=True, v2_pdf=True, v3_status=None, v3_updated=None, v3_json=True, v3_xml=True):
        super(DiscoveryList, self).__init__(versions={'values': []})
        href = href or self.TEST_URL
        if v2:
            v2_href = href.rstrip('/') + '/v2.0'
            self.add_v2(v2_href, id=v2_id, status=v2_status, updated=v2_updated, html=v2_html, pdf=v2_pdf)
        if v3:
            v3_href = href.rstrip('/') + '/v3'
            self.add_v3(v3_href, id=v3_id, status=v3_status, updated=v3_updated, json=v3_json, xml=v3_xml)

    @property
    def versions(self):
        return self['versions']['values']

    def add_version(self, version):
        """Add a new version structure to the list.

        :param dict version: A new version structure to add to the list.
        """
        self.versions.append(version)

    def add_v2(self, href, **kwargs):
        """Add a v2 version to the list.

        The parameters are the same as V2Discovery.
        """
        obj = V2Discovery(href, **kwargs)
        self.add_version(obj)
        return obj

    def add_v3(self, href, **kwargs):
        """Add a v3 version to the list.

        The parameters are the same as V3Discovery.
        """
        obj = V3Discovery(href, **kwargs)
        self.add_version(obj)
        return obj

    def add_microversion(self, href, id, **kwargs):
        """Add a microversion version to the list.

        The parameters are the same as MicroversionDiscovery.
        """
        obj = MicroversionDiscovery(href=href, id=id, **kwargs)
        self.add_version(obj)
        return obj

    def add_nova_microversion(self, href, id, **kwargs):
        """Add a nova microversion version to the list.

        The parameters are the same as NovaMicroversionDiscovery.
        """
        obj = NovaMicroversionDiscovery(href=href, id=id, **kwargs)
        self.add_version(obj)
        return obj