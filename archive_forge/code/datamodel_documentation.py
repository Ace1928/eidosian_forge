from kivy.properties import ListProperty, ObservableDict, ObjectProperty
from kivy.event import EventDispatcher
from functools import partial
A dictionary instance, which when modified will trigger a `data` and
        consequently an `on_data_changed` dispatch.
        