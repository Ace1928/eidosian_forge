from functools import partial
import json
import logging
import click
import cligj
from fiona.fio import helpers, options, with_context_env
from fiona.model import Geometry, ObjectEncoder
from fiona.transform import transform_geom
Make a GeoJSON feature collection from a sequence of GeoJSON
    features and print it.