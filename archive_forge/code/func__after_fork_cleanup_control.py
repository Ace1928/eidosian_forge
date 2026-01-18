import warnings
from billiard.common import TERM_SIGNAME
from kombu.matcher import match
from kombu.pidbox import Mailbox
from kombu.utils.compat import register_after_fork
from kombu.utils.functional import lazy
from kombu.utils.objects import cached_property
from celery.exceptions import DuplicateNodenameWarning
from celery.utils.log import get_logger
from celery.utils.text import pluralize
def _after_fork_cleanup_control(control):
    try:
        control._after_fork()
    except Exception as exc:
        logger.info('after fork raised exception: %r', exc, exc_info=1)