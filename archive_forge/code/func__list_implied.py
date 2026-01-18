import logging
from osc_lib.command import command
from openstackclient.i18n import _
def _list_implied(response):
    for rule in response:
        for implies in rule.implies:
            yield (rule.prior_role['id'], rule.prior_role['name'], implies['id'], implies['name'])