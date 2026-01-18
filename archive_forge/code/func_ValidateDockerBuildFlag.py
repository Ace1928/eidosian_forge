from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.third_party.appengine.api import appinfo
def ValidateDockerBuildFlag(unused_value):
    raise argparse.ArgumentTypeError("The --docker-build flag no longer exists.\n\nDocker images are now built remotely using Google Container Builder. To run a\nDocker build on your own host, you can run:\n  docker build -t gcr.io/<project>/<service.version> .\n  gcloud docker push gcr.io/<project>/<service.version>\n  gcloud app deploy --image-url=gcr.io/<project>/<service.version>\nIf you don't already have a Dockerfile, you must run:\n  gcloud beta app gen-config\nfirst to get one.\n  ")