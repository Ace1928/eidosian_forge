import logging
from pyomo.common.collections import ComponentMap
from pyomo.contrib.viewer.qt import *
import pyomo.environ as pyo
class UIDataNoUi(object):
    """
    This is the UIData object minus the signals.  This is the base class for
    UIData.  The class is split this way for testing when PyQt is not available.
    """

    def __init__(self, model=None):
        """
        This class holds the basic UI setup, but doesn't depend on Qt. It
        shouldn't really be used except for testing when Qt is not available.

        Args:
            model: The Pyomo model to view
        """
        super().__init__()
        self._model = None
        self._begin_update = False
        self.value_cache = ComponentMap()
        self.value_cache_units = ComponentMap()
        self.begin_update()
        self.model = model
        self.end_update()

    def begin_update(self):
        """
        Lets the model setup be changed without emitting the updated signal
        until the end_update function is called.
        """
        self._begin_update = True

    def end_update(self, emit=True):
        """
        Sets the begin update flag to false.  Needs to be overloaded to also
        emit an update signal in the full UIData class
        """
        self._begin_update = False

    def emit_update(self):
        """
        Don't forget to overloaded this, not raising a NotImplementedError so
        tests can run without Qt
        """
        pass

    def emit_exec_refresh(self):
        """
        Don't forget to overloaded this, not raising a NotImplementedError so
        tests can run without Qt
        """
        pass

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.value_cache = ComponentMap()
        self.value_cache_units = ComponentMap()
        self.emit_update()

    def calculate_constraints(self):
        for o in self.model.component_data_objects(pyo.Constraint, active=True):
            try:
                self.value_cache[o] = pyo.value(o.body, exception=False)
            except ZeroDivisionError:
                self.value_cache[o] = 'Divide_by_0'
        self.emit_exec_refresh()

    def calculate_expressions(self):
        for o in self.model.component_data_objects(pyo.Expression, active=True):
            try:
                self.value_cache[o] = pyo.value(o, exception=False)
            except ZeroDivisionError:
                self.value_cache[o] = 'Divide_by_0'
            try:
                self.value_cache_units[o] = str(pyo.units.get_units(o))
            except:
                pass
        self.emit_exec_refresh()