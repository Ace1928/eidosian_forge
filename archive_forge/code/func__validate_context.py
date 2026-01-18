import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _validate_context(self, ctx):
    if type(ctx).__name__ != 'RequestContext':
        raise TypeError('ctx must be a RequestContext; got: %r' % (ctx,))