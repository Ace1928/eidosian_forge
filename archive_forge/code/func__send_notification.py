import abc
import argparse
import logging
import uuid
from oslo_config import cfg
from oslo_utils import timeutils
from stevedore import extension
from stevedore import named
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import transport as msg_transport
def _send_notification():
    """Command line tool to send notifications manually."""
    parser = argparse.ArgumentParser(description='Oslo.messaging notification sending')
    parser.add_argument('--config-file', help='Path to configuration file')
    parser.add_argument('--transport-url', help='Transport URL')
    parser.add_argument('--publisher-id', help='Publisher ID')
    parser.add_argument('--event-type', default='test', help='Event type')
    parser.add_argument('--topic', nargs='*', help='Topic to send to')
    parser.add_argument('--priority', default='info', choices=('info', 'audit', 'warn', 'error', 'critical', 'sample'), help='Event type')
    parser.add_argument('--driver', default='messagingv2', choices=extension.ExtensionManager('oslo.messaging.notify.drivers').names(), help='Notification driver')
    parser.add_argument('payload', help='the notification payload (dict)')
    args = parser.parse_args()
    conf = cfg.ConfigOpts()
    conf([], default_config_files=[args.config_file] if args.config_file else None)
    transport = get_notification_transport(conf, url=args.transport_url)
    notifier = Notifier(transport, args.publisher_id, topics=args.topic, driver=args.driver)
    notifier._notify({}, args.event_type, args.payload, args.priority)