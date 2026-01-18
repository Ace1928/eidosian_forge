import threading
import json
from gc import get_objects, garbage
from kivy.clock import Clock
from kivy.cache import Cache
from collections import OrderedDict
from kivy.logger import Logger
class FlaskThread(threading.Thread):

    def run(self):
        Clock.schedule_interval(self.dump_metrics, 0.1)
        app.run(debug=True, use_debugger=True, use_reloader=False)

    def dump_metrics(self, dt):
        m = metrics
        m['Python objects'].append(len(get_objects()))
        m['Python garbage'].append(len(garbage))
        m['FPS (internal)'].append(Clock.get_fps())
        m['FPS (real)'].append(Clock.get_rfps())
        m['Events'].append(len(Clock.get_events()))
        for category in Cache._categories:
            m['Cache ' + category].append(len(Cache._objects.get(category, [])))
        for values in m.values():
            values.pop(0)
            values[0] = 0