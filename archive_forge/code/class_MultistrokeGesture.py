import pickle
import base64
import zlib
from re import match as re_match
from collections import deque
from math import sqrt, pi, radians, acos, atan, atan2, pow, floor
from math import sin as math_sin, cos as math_cos
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.properties import ListProperty
from kivy.compat import PY2
from io import BytesIO
class MultistrokeGesture(object):
    """:class:`MultistrokeGesture` represents a gesture. It maintains a set of
    `strokes` and generates unistroke (ie :class:`UnistrokeTemplate`)
    permutations that are used for evaluating candidates against this gesture
    later.

    :Arguments:
        `name`
            Identifies the name of the gesture - it is returned to you in the
            results of a :meth:`Recognizer.recognize` search. You can have any
            number of MultistrokeGesture objects with the same name; many
            definitions of one gesture. The same name is given to all the
            generated unistroke permutations. Required, no default.
        `strokes`
            A list of paths that represents the gesture. A path is a list of
            Vector objects::

                gesture = MultistrokeGesture('my_gesture', strokes=[
                  [Vector(x1, y1), Vector(x2, y2), ...... ], # stroke 1
                  [Vector(), Vector(), Vector(), Vector() ]  # stroke 2
                  #, [stroke 3], [stroke 4], ...
                ])

            For template matching purposes, all the strokes are combined to a
            single list (unistroke). You should still specify the strokes
            individually, and set `stroke_sensitive` True (whenever possible).

            Once you do this, unistroke permutations are immediately generated
            and stored in `self.templates` for later, unless you set the
            `permute` flag to False.
        `priority`
            Determines when :func:`Recognizer.recognize` will attempt to match
            this template, lower priorities are evaluated first (only if
            a priority `filter` is used). You should use lower priority on
            gestures that are more likely to match. For example, set user
            templates at lower number than generic templates. Default is 100.
        `numpoints`
            Determines the number of points this gesture should be resampled to
            (for matching purposes). The default is 16.
        `stroke_sensitive`
            Determines if the number of strokes (paths) in this gesture is
            required to be the same in the candidate (user input) gesture
            during matching. If this is False, candidates will always be
            evaluated, disregarding the number of strokes. Default is True.
        `orientation_sensitive`
            Determines if this gesture is orientation sensitive. If True,
            aligns the indicative orientation with the one of eight base
            orientations that requires least rotation. Default is True.
        `angle_similarity`
            This is used by the :func:`Recognizer.recognize` function when a
            candidate is evaluated against this gesture. If the angles between
            them are too far off, the template is considered a non-match.
            Default is 30.0 (degrees)
        `permute`
            If False, do not use Heap Permute algorithm to generate different
            stroke orders when instantiated. If you set this to False, a
            single UnistrokeTemplate built from `strokes` is used.
    """

    def __init__(self, name, strokes=None, **kwargs):
        self.name = name
        self.priority = kwargs.get('priority', 100)
        self.numpoints = kwargs.get('numpoints', 16)
        self.stroke_sens = kwargs.get('stroke_sensitive', True)
        self.orientation_sens = kwargs.get('orientation_sensitive', True)
        self.angle_similarity = kwargs.get('angle_similarity', 30.0)
        self.strokes = []
        if strokes is not None:
            self.strokes = strokes
            if kwargs.get('permute', True):
                self.permute()
            else:
                self.templates = [UnistrokeTemplate(name, points=[i for sub in strokes for i in sub], numpoints=self.numpoints, orientation_sensitive=self.orientation_sens)]

    def angle_similarity_threshold(self):
        return radians(self.angle_similarity)

    def add_stroke(self, stroke, permute=False):
        """Add a stroke to the self.strokes list. If `permute` is True, the
        :meth:`permute` method is called to generate new unistroke templates"""
        self.strokes.append(stroke)
        if permute:
            self.permute()

    def get_distance(self, cand, tpl, numpoints=None):
        """Compute the distance from this Candidate to a UnistrokeTemplate.
        Returns the Cosine distance between the stroke paths.

        `numpoints` will prepare both the UnistrokeTemplate and Candidate path
        to n points (when necessary), you probably don't want to do this.
        """
        n = numpoints
        if n is None or n < 2:
            n = self.numpoints
        v1 = tpl.get_vector(n)
        v2 = cand.get_protractor_vector(n, tpl.orientation_sens)
        a = 0.0
        b = 0.0
        for i in xrange(0, len(v1), 2):
            a += v1[i] * v2[i] + v1[i + 1] * v2[i + 1]
            b += v1[i] * v2[i + 1] - v1[i + 1] * v2[i]
        angle = atan(b / a)
        result = a * math_cos(angle) + b * math_sin(angle)
        if result >= 1:
            result = 1
        elif result <= -1:
            result = -1
        return acos(result)

    def match_candidate(self, cand, **kwargs):
        """Match a given candidate against this MultistrokeGesture object. Will
        test against all templates and report results as a list of four
        items:

            `index 0`
                Best matching template's index (in self.templates)
            `index 1`
                Computed distance from the template to the candidate path
            `index 2`
                List of distances for all templates. The list index
                corresponds to a :class:`UnistrokeTemplate` index in
                self.templates.
            `index 3`
                Counter for the number of performed matching operations, ie
                templates matched against the candidate
        """
        best_d = float('infinity')
        best_tpl = None
        mos = 0
        out = []
        if self.stroke_sens and len(self.strokes) != len(cand.strokes):
            return (best_tpl, best_d, out, mos)
        skip_bounded = cand.skip_bounded
        skip_invariant = cand.skip_invariant
        get_distance = self.get_distance
        ang_sim_threshold = self.angle_similarity_threshold()
        for idx, tpl in enumerate(self.templates):
            if tpl.orientation_sens:
                if skip_bounded:
                    continue
            elif skip_invariant:
                continue
            mos += 1
            n = kwargs.get('force_numpoints', tpl.numpoints)
            ang_sim = cand.get_angle_similarity(tpl, numpoints=n)
            if ang_sim > ang_sim_threshold:
                continue
            d = get_distance(cand, tpl, numpoints=n)
            out.append(d)
            if d < best_d:
                best_d = d
                best_tpl = idx
        return (best_tpl, best_d, out, mos)

    def permute(self):
        """Generate all possible unistroke permutations from self.strokes and
        save the resulting list of UnistrokeTemplate objects in self.templates.

        Quote from http://faculty.washington.edu/wobbrock/pubs/gi-10.2.pdf ::

            We use Heap Permute [16] (p. 179) to generate all stroke orders
            in a multistroke gesture. Then, to generate stroke directions for
            each order, we treat each component stroke as a dichotomous
            [0,1] variable. There are 2^N combinations for N strokes, so we
            convert the decimal values 0 to 2^N-1, inclusive, to binary
            representations and regard each bit as indicating forward (0) or
            reverse (1). This algorithm is often used to generate truth tables
            in propositional logic.

        See section 4.1: "$N Algorithm" of the linked paper for details.

        .. Warning ::

            Using heap permute for gestures with more than 3 strokes
            can result in very large number of templates (a 9-stroke
            gesture = 38 million templates). If you are dealing with
            these types of gestures, you should manually compose
            all the desired stroke orders.
        """
        self._order = [i for i in xrange(0, len(self.strokes))]
        self._orders = []
        self._heap_permute(len(self.strokes))
        del self._order
        self.templates = [UnistrokeTemplate(self.name, points=permutation, numpoints=self.numpoints, orientation_sensitive=self.orientation_sens) for permutation in self._make_unistrokes()]
        del self._orders

    def _heap_permute(self, n):
        self_order = self._order
        if n == 1:
            self._orders.append(self_order[:])
        else:
            i = 0
            for i in xrange(0, n):
                self._heap_permute(n - 1)
                if n % 2 == 1:
                    tmp = self_order[0]
                    self_order[0] = self_order[n - 1]
                    self_order[n - 1] = tmp
                else:
                    tmp = self_order[i]
                    self_order[i] = self_order[n - 1]
                    self_order[n - 1] = tmp

    def _make_unistrokes(self):
        unistrokes = []
        unistrokes_append = unistrokes.append
        self_strokes = self.strokes
        for r in self._orders:
            b = 0
            while b < pow(2, len(r)):
                unistroke = []
                unistroke_append = unistroke.append
                for i in xrange(0, len(r)):
                    pts = self_strokes[r[i]][:]
                    if b >> i & 1 == 1:
                        pts.reverse()
                    unistroke_append(None)
                    unistroke[-1:] = pts
                unistrokes_append(unistroke)
                b += 1
        return unistrokes