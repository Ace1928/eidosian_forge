import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
class Gesture:
    """A python implementation of a gesture recognition algorithm by
    Oleg Dopertchouk: http://www.gamedev.net/reference/articles/article2039.asp

    Implemented by Jeiel Aranal (chemikhazi@gmail.com),
    released into the public domain.
    """
    DEFAULT_TOLERANCE = 0.1

    def __init__(self, tolerance=None):
        """
        Gesture([tolerance=float])
        Creates a new gesture with an optional matching tolerance value.
        """
        self.width = 0.0
        self.height = 0.0
        self.gesture_product = 0.0
        self.strokes = list()
        if tolerance is None:
            self.tolerance = Gesture.DEFAULT_TOLERANCE
        else:
            self.tolerance = tolerance

    def _scale_gesture(self):
        """ Scales down the gesture to a unit of 1."""
        min_x = min([stroke.min_x for stroke in self.strokes])
        max_x = max([stroke.max_x for stroke in self.strokes])
        min_y = min([stroke.min_y for stroke in self.strokes])
        max_y = max([stroke.max_y for stroke in self.strokes])
        x_len = max_x - min_x
        self.width = x_len
        y_len = max_y - min_y
        self.height = y_len
        scale_factor = max(x_len, y_len)
        if scale_factor <= 0.0:
            return False
        scale_factor = 1.0 / scale_factor
        for stroke in self.strokes:
            stroke.scale_stroke(scale_factor)
        return True

    def _center_gesture(self):
        """ Centers the Gesture.points of the gesture."""
        total_x = 0.0
        total_y = 0.0
        total_points = 0
        for stroke in self.strokes:
            stroke_y = sum([pt.y for pt in stroke.points])
            stroke_x = sum([pt.x for pt in stroke.points])
            total_y += stroke_y
            total_x += stroke_x
            total_points += len(stroke.points)
        if total_points == 0:
            return False
        total_x /= total_points
        total_y /= total_points
        for stroke in self.strokes:
            stroke.center_stroke(total_x, total_y)
        return True

    def add_stroke(self, point_list=None):
        """Adds a stroke to the gesture and returns the Stroke instance.
           Optional point_list argument is a list of the mouse points for
           the stroke.
        """
        self.strokes.append(GestureStroke())
        if isinstance(point_list, list) or isinstance(point_list, tuple):
            for point in point_list:
                if isinstance(point, GesturePoint):
                    self.strokes[-1].points.append(point)
                elif isinstance(point, list) or isinstance(point, tuple):
                    if len(point) != 2:
                        raise ValueError('Stroke entry must have 2 values max')
                    self.strokes[-1].add_point(point[0], point[1])
                else:
                    raise TypeError('The point list should either be tuples of x and y or a list of GesturePoint objects')
        elif point_list is not None:
            raise ValueError('point_list should be a tuple/list')
        return self.strokes[-1]

    def normalize(self, stroke_samples=32):
        """Runs the gesture normalization algorithm and calculates the dot
        product with self.
        """
        if not self._scale_gesture() or not self._center_gesture():
            self.gesture_product = False
            return False
        for stroke in self.strokes:
            stroke.normalize_stroke(stroke_samples)
        self.gesture_product = self.dot_product(self)

    def get_rigid_rotation(self, dstpts):
        """
        Extract the rotation to apply to a group of points to minimize the
        distance to a second group of points. The two groups of points are
        assumed to be centered. This is a simple version that just picks
        an angle based on the first point of the gesture.
        """
        if len(self.strokes) < 1 or len(self.strokes[0].points) < 1:
            return 0
        if len(dstpts.strokes) < 1 or len(dstpts.strokes[0].points) < 1:
            return 0
        p = dstpts.strokes[0].points[0]
        target = Vector([p.x, p.y])
        source = Vector([p.x, p.y])
        return source.angle(target)

    def dot_product(self, comparison_gesture):
        """ Calculates the dot product of the gesture with another gesture."""
        if len(comparison_gesture.strokes) != len(self.strokes):
            return -1
        if getattr(comparison_gesture, 'gesture_product', True) is False or getattr(self, 'gesture_product', True) is False:
            return -1
        dot_product = 0.0
        for stroke_index, (my_stroke, cmp_stroke) in enumerate(list(zip(self.strokes, comparison_gesture.strokes))):
            for pt_index, (my_point, cmp_point) in enumerate(list(zip(my_stroke.points, cmp_stroke.points))):
                dot_product += my_point.x * cmp_point.x + my_point.y * cmp_point.y
        return dot_product

    def rotate(self, angle):
        g = Gesture()
        for stroke in self.strokes:
            tmp = []
            for j in stroke.points:
                v = Vector([j.x, j.y]).rotate(angle)
                tmp.append(v)
            g.add_stroke(tmp)
        g.gesture_product = g.dot_product(g)
        return g

    def get_score(self, comparison_gesture, rotation_invariant=True):
        """ Returns the matching score of the gesture against another gesture.
        """
        if isinstance(comparison_gesture, Gesture):
            if rotation_invariant:
                angle = self.get_rigid_rotation(comparison_gesture)
                comparison_gesture = comparison_gesture.rotate(angle)
            score = self.dot_product(comparison_gesture)
            if score <= 0:
                return score
            score /= math.sqrt(self.gesture_product * comparison_gesture.gesture_product)
            return score

    def __eq__(self, comparison_gesture):
        """ Allows easy comparisons between gesture instances."""
        if isinstance(comparison_gesture, Gesture):
            score = self.get_score(comparison_gesture)
            if score > 1.0 - self.tolerance and score < 1.0 + self.tolerance:
                return True
            else:
                return False
        else:
            return NotImplemented

    def __ne__(self, comparison_gesture):
        result = self.__eq__(comparison_gesture)
        if result is NotImplemented:
            return result
        else:
            return not result

    def __lt__(self, comparison_gesture):
        raise TypeError('Gesture cannot be evaluated with <')

    def __gt__(self, comparison_gesture):
        raise TypeError('Gesture cannot be evaluated with >')

    def __le__(self, comparison_gesture):
        raise TypeError('Gesture cannot be evaluated with <=')

    def __ge__(self, comparison_gesture):
        raise TypeError('Gesture cannot be evaluated with >=')