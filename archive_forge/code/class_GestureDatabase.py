import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
class GestureDatabase(object):
    """Class to handle a gesture database."""

    def __init__(self):
        self.db = []

    def add_gesture(self, gesture):
        """Add a new gesture to the database."""
        self.db.append(gesture)

    def find(self, gesture, minscore=0.9, rotation_invariant=True):
        """Find a matching gesture in the database."""
        if not gesture:
            return
        best = None
        bestscore = minscore
        for g in self.db:
            score = g.get_score(gesture, rotation_invariant)
            if score < bestscore:
                continue
            bestscore = score
            best = g
        if not best:
            return
        return (bestscore, best)

    def gesture_to_str(self, gesture):
        """Convert a gesture into a unique string."""
        io = BytesIO()
        p = pickle.Pickler(io)
        p.dump(gesture)
        data = base64.b64encode(zlib.compress(io.getvalue(), 9))
        return data

    def str_to_gesture(self, data):
        """Convert a unique string to a gesture."""
        io = BytesIO(zlib.decompress(base64.b64decode(data)))
        p = pickle.Unpickler(io)
        gesture = p.load()
        return gesture