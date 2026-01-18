from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import zipfile
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files
from six.moves import urllib
class ListArchives:
    """Adds additional helpful fields to a list of archives."""

    def __init__(self, org):
        self._org = org
        self._lro_helper = apigee.LROPoller(org)
        self._deployed_status = 'Deployed'
        self._inprogress_status = 'In Progress'
        self._failed_status = 'Failed'
        self._not_found_status = 'Not Found'
        self._unknown_status = 'Unknown'
        self._missing_status = 'Missing'

    def ExtendedArchives(self, archives):
        """Given a list of archives, extends them with a status and error field where needed.

    Args:
      archives: A list of archives to extend with a status and potential error.

    Returns:
      A list of archives with their associated status.
    """
        extended_archives = sorted(archives, key=lambda k: k['createdAt'], reverse=True)
        cascade_unknown = False
        for idx, a in enumerate(extended_archives):
            serilized_archive = resource_projector.MakeSerializable(a)
            if cascade_unknown:
                serilized_archive['operationStatus'] = self._unknown_status
            elif 'operation' in a:
                uuid = apigee.OperationsClient.SplitName({'name': a['operation']})['uuid']
                try:
                    op = apigee.OperationsClient.Describe({'organizationsId': self._org, 'operationsId': uuid})
                    status = self._StatusFromOperation(op)
                    serilized_archive['operationStatus'] = status['status']
                    if status['status'] == self._deployed_status:
                        extended_archives[idx] = serilized_archive
                        return extended_archives
                    elif 'error' in status:
                        serilized_archive['error'] = status['error']
                except errors.EntityNotFoundError:
                    serilized_archive['operationStatus'] = self._not_found_status
                except:
                    cascade_unknown = True
                    serilized_archive['operationStatus'] = self._unknown_status
            else:
                serilized_archive['operationStatus'] = self._missing_status
            extended_archives[idx] = serilized_archive
        return extended_archives

    def _StatusFromOperation(self, op):
        """Gathers given an LRO, determines the associated archive status.

    Args:
      op: An Apigee LRO

    Returns:
      A dict in the format of
        {"status": "{status}", "error": "{error if present on LRO}"}.
    """
        status = {}
        try:
            is_done = self._lro_helper.IsDone(op)
            if is_done:
                status['status'] = self._deployed_status
            else:
                status['status'] = self._inprogress_status
        except errors.RequestError:
            status['status'] = self._failed_status
            status['error'] = op['error']['message']
        return status