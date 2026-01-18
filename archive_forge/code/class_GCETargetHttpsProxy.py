import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class GCETargetHttpsProxy(UuidMixin):
    """GCETargetHttpsProxy represents the TargetHttpsProxy resource."""

    def __init__(self, id, name, description=None, sslcertificates=None, urlmap=None, driver=None, extra=None):
        """
        :param  name:  Name of the resource. Provided by the client when the
                       resource is created. The name must be 1-63 characters
                       long, and comply with RFC1035. Specifically, the name
                       must be 1-63 characters long and match the regular
                       expression [a-z]([-a-z0-9]*[a-z0-9])? which means the
                       first character must be a lowercase letter, and all
                       following characters must be a dash, lowercase letter,
                       or digit, except the last character, which cannot be a
                       dash.
        :type   name: ``str``

        :param  description:  An optional description of this resource.
                              Provide this property when you create the
                              resource.
        :type   description: ``str``

        :param  sslcertificates:  URLs to SslCertificate resources that are
                                   used to authenticate connections between
                                   users and the load balancer. Currently,
                                   exactly one SSL certificate must be
                                   specified.
        :type   sslcertificates: ``list`` of :class:`GCESslcertificates`

        :param  urlmap:  A fully-qualified or valid partial URL to the
                          UrlMap resource that defines the mapping from URL
                          to the BackendService. For example, the following
                          are all valid URLs for specifying a URL map:   - ht
                          tps://www.googleapis.compute/v1/projects/project/gl
                          obal/urlMaps/url-map  -
                          projects/project/global/urlMaps/url-map  -
                          global/urlMaps/url-map
        :type   urlmap: :class:`GCEUrlMap`

        :keyword  driver:  An initialized :class: `GCENodeDriver`
        :type   driver: :class:`:class: `GCENodeDriver``

        :keyword  extra:  A dictionary of extra information.
        :type   extra: ``:class: ``dict````

        """
        self.name = name
        self.description = description
        self.sslcertificates = sslcertificates
        self.urlmap = urlmap
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def __repr__(self):
        return '<GCETargetHttpsProxy name="%s">' % self.name

    def set_sslcertificates(self, sslcertificates):
        """
        Set the SSL Certificates for this TargetHTTPSProxy

        :param  sslcertificates: SSL Certificates to set.
        :type   sslcertificates: ``list`` of :class:`GCESslCertificate`

        :return:  True if successful
        :rtype:   ``bool``
        """
        return self.driver.ex_targethttpsproxy_set_sslcertificates(targethttpsproxy=self, sslcertificates=sslcertificates)

    def set_urlmap(self, urlmap):
        """
        Changes the URL map for TargetHttpsProxy.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  targethttpsproxy:  Name of the TargetHttpsProxy resource
                                   whose URL map is to be set.
        :type   targethttpsproxy: ``str``

        :param  urlmap:  UrlMap to set.
        :type   urlmap: :class:`GCEUrlMap`

        :return:  True
        :rtype: ``bool``
        """
        return self.driver.ex_targethttpsproxy_set_urlmap(targethttpsproxy=self, urlmap=urlmap)

    def destroy(self):
        """
        Destroy this TargetHttpsProxy.

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_destroy_targethttpsproxy(targethttpsproxy=self)