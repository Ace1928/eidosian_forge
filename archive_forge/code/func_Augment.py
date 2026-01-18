from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import argparse
import collections
from collections.abc import Collection, Container, Iterable, Mapping, MutableMapping
import copy
import dataclasses
import itertools
import json
import types
from typing import Any, ClassVar
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
import six
def Augment(instance_str):
    instance = instance_str.split(':')
    if len(instance) == 3:
        return ':'.join(instance)
    elif len(instance) == 1:
        if not project:
            raise exceptions.CloudSQLError('To specify a Cloud SQL instance by plain name, you must specify a project.')
        if not region:
            raise exceptions.CloudSQLError('To specify a Cloud SQL instance by plain name, you must be deploying to a managed Cloud Run region.')
        return ':'.join(itertools.chain([project, region], instance))
    else:
        raise exceptions.CloudSQLError('Malformed CloudSQL instance string: {}'.format(instance_str))