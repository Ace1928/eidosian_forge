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
def _recognize_tick(dt):
    start_gc = result._completed
    stop_now = False
    while not stop_now and (tasklist and (not result._break_flag)) and (not GPF or result._completed - start_gc < GPF):
        if timeout and Clock.get_time() - result._start_time >= timeout:
            result.status = 'timeout'
            stop_now = True
            break
        gesture = tasklist.popleft()
        tpl, d, res, mos = gesture.match_candidate(cand, **kwargs)
        if tpl is not None:
            score = result._add_result(gesture, d, tpl, res)
            if goodscore is not None and score >= goodscore:
                result.status = 'goodscore'
                stop_now = True
        result._match_ops += mos
        result._completed += 1
        result.dispatch('on_progress')

    def _dispatch():
        result.dispatch('on_complete')
        self.dispatch('on_search_complete', result)
        return False
    if not tasklist:
        result.status = 'complete'
        return _dispatch()
    elif result._break_flag:
        result.status = 'stop'
        return _dispatch()
    elif stop_now:
        return _dispatch()
    else:
        Clock.schedule_once(_recognize_tick, delay)
        return True