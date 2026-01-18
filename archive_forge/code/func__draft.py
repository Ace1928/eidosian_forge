import zmq
from zmq.backend import Frame as FrameBase
from .attrsettr import AttributeSetter
def _draft(v, feature):
    zmq.error._check_version(v, feature)
    if not zmq.DRAFT_API:
        raise RuntimeError('libzmq and pyzmq must be built with draft support for %s' % feature)