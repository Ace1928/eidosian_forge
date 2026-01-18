import atexit
import os
import sys
import __main__
from contextlib import suppress
from io import BytesIO
import dill
import json                                         # top-level module
import urllib as url                                # top-level module under alias
from xml import sax                                 # submodule
import xml.dom.minidom as dom                       # submodule under alias
import test_dictviews as local_mod                  # non-builtin top-level module
from calendar import Calendar, isleap, day_name     # class, function, other object
from cmath import log as complex_log                # imported with alias
def _test_objects(main, globals_copy, refimported):
    try:
        main_dict = __main__.__dict__
        global Person, person, Calendar, CalendarSubclass, cal, selfref
        for obj in ('json', 'url', 'local_mod', 'sax', 'dom'):
            assert globals()[obj].__name__ == globals_copy[obj].__name__
        for obj in ('x', 'empty', 'names'):
            assert main_dict[obj] == globals_copy[obj]
        for obj in ['squared', 'cubed']:
            assert main_dict[obj].__globals__ is main_dict
            assert main_dict[obj](3) == globals_copy[obj](3)
        assert Person.__module__ == __main__.__name__
        assert isinstance(person, Person)
        assert person.age == globals_copy['person'].age
        assert issubclass(CalendarSubclass, Calendar)
        assert isinstance(cal, CalendarSubclass)
        assert cal.weekdays() == globals_copy['cal'].weekdays()
        assert selfref is __main__
    except AssertionError as error:
        error.args = (_error_line(error, obj, refimported),)
        raise