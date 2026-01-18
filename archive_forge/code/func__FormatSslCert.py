from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def _FormatSslCert(self, cert_id):
    res = self._registry.Parse(cert_id, params={'appsId': self.project}, collection='appengine.apps.authorizedCertificates')
    return res.RelativeName()