from collections import abc
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import opts
from oslo_policy import policy
from glance.common import exception
from glance.domain import proxy
from glance import policies
def _enforce_image_visibility(policy, context, visibility, target):
    if visibility == 'public':
        policy.enforce(context, 'publicize_image', target)
    elif visibility == 'community':
        policy.enforce(context, 'communitize_image', target)