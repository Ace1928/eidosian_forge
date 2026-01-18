from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.distribution import compressor
from tensorboard.plugins.distribution import metadata
from tensorboard.plugins.histogram import histograms_plugin
Given a tag and single run, return an array of compressed
        histograms.