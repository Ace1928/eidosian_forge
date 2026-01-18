import pytest
import sys
from pathlib import Path
import catalogue
def _check_entry_points():
    assert catalogue.REGISTRY == {}
    test_registry = catalogue.create('test', 'foo', entry_points=True)
    entry_points = test_registry.get_entry_points()
    assert 'bar' in entry_points
    assert entry_points['bar'] == catalogue.check_exists
    assert test_registry.get_entry_point('bar') == catalogue.check_exists
    assert catalogue.REGISTRY == {}
    assert test_registry.get('bar') == catalogue.check_exists
    assert test_registry.get_all() == {'bar': catalogue.check_exists}
    assert 'bar' in test_registry