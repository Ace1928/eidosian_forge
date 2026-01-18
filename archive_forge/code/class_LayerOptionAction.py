import argparse
from django.contrib.gis import gdal
from django.core.management.base import BaseCommand, CommandError
from django.utils.inspect import get_func_args
class LayerOptionAction(argparse.Action):
    """
    Custom argparse action for the `ogrinspect` `layer_key` keyword option
    which may be an integer or a string.
    """

    def __call__(self, parser, namespace, value, option_string=None):
        try:
            setattr(namespace, self.dest, int(value))
        except ValueError:
            setattr(namespace, self.dest, value)