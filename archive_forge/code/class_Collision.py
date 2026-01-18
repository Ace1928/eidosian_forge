from eventlet.event import Event
from eventlet import greenthread
import collections
class Collision(Exception):
    """
    DAGPool raises Collision when you try to launch two greenthreads with the
    same key, or post() a result for a key corresponding to a greenthread, or
    post() twice for the same key. As with KeyError, str(collision) names the
    key in question.
    """
    pass