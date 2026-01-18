from __future__ import absolute_import
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.iamcredentials_apitools.iamcredentials_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
class ProjectsService(base_api.BaseApiService):
    """Service class for the projects resource."""
    _NAME = u'projects'

    def __init__(self, client):
        super(IamcredentialsV1.ProjectsService, self).__init__(client)
        self._upload_configs = {}