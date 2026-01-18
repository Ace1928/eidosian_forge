import logging
import os
import sys
import warnings
from cliff import app
from cliff import commandmanager
from keystoneauth1 import loading
from oslo_utils import importutils
from vitrageclient import __version__
from vitrageclient import auth
from vitrageclient import client
from vitrageclient.v1.cli import alarm
from vitrageclient.v1.cli import event
from vitrageclient.v1.cli import healthcheck
from vitrageclient.v1.cli import rca
from vitrageclient.v1.cli import resource
from vitrageclient.v1.cli import service
from vitrageclient.v1.cli import status
from vitrageclient.v1.cli import template
from vitrageclient.v1.cli import topology
from vitrageclient.v1.cli import webhook
class VitrageCommandManager(commandmanager.CommandManager):
    COMMANDS = {'topology show': topology.TopologyShow, 'resource show': resource.ResourceShow, 'resource list': resource.ResourceList, 'resource count': resource.ResourceCount, 'alarm list': alarm.AlarmList, 'alarm history': alarm.AlarmHistory, 'alarm show': alarm.AlarmShow, 'alarm count': alarm.AlarmCount, 'rca show': rca.RcaShow, 'template validate': template.TemplateValidate, 'template list': template.TemplateList, 'template versions': template.TemplateVersions, 'template show': template.TemplateShow, 'template add': template.TemplateAdd, 'template delete': template.TemplateDelete, 'event post': event.EventPost, 'healthcheck': healthcheck.HealthCheck, 'webhook delete': webhook.WebhookDelete, 'webhook add': webhook.WebhookAdd, 'webhook list': webhook.WebhookList, 'webhook show': webhook.WebhookShow, 'service list': service.ServiceList, 'status': status.Status}

    def load_commands(self, namespace):
        for k, v in self.COMMANDS.items():
            self.add_command(k, v)