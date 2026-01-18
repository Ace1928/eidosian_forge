import os
import pkg_resources
from paste.script import pluginlib, copydir
from paste.script.command import BadCommand
import subprocess

        Runs the command, respecting verbosity and simulation.
        Returns stdout, or None if simulating.
        