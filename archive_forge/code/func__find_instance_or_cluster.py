import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _find_instance_or_cluster(cs, instance_or_cluster):
    """Returns an instance or cluster, found by id, along with the type of
    resource, instance or cluster, that was found.
    Raises CommandError if none is found.
    """
    try:
        return (_find_instance(cs, instance_or_cluster), 'instance')
    except exceptions.CommandError:
        try:
            return (_find_cluster(cs, instance_or_cluster), 'cluster')
        except Exception:
            raise exceptions.CommandError(_("No instance or cluster with a name or ID of '%s' exists.") % instance_or_cluster)