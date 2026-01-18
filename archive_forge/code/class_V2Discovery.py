from keystoneauth1 import _utils as utils
class V2Discovery(DiscoveryBase):
    """A Version element for a V2 identity service endpoint.

    Provides some default values and helper methods for creating a v2.0
    endpoint version structure. Clients should use this instead of creating
    their own structures.

    :param string href: The url that this entry should point to.
    :param string id: The version id that should be reported. (optional)
                      Defaults to 'v2.0'.
    :param bool html: Add HTML describedby links to the structure.
    :param bool pdf: Add PDF describedby links to the structure.

    """
    _DESC_URL = 'https://developer.openstack.org/api-ref/identity/v2/'

    def __init__(self, href, id=None, html=True, pdf=True, **kwargs):
        super(V2Discovery, self).__init__(id or 'v2.0', **kwargs)
        self.add_link(href)
        if html:
            self.add_html_description()
        if pdf:
            self.add_pdf_description()

    def add_html_description(self):
        """Add the HTML described by links.

        The standard structure includes a link to a HTML document with the
        API specification. Add it to this entry.
        """
        self.add_link(href=self._DESC_URL + 'content', rel='describedby', type='text/html')

    def add_pdf_description(self):
        """Add the PDF described by links.

        The standard structure includes a link to a PDF document with the
        API specification. Add it to this entry.
        """
        self.add_link(href=self._DESC_URL + 'identity-dev-guide-2.0.pdf', rel='describedby', type='application/pdf')