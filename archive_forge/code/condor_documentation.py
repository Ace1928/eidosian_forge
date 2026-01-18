import os
from time import sleep
from ...interfaces.base import CommandLine
from ... import logging
from .base import SGELikeBatchManagerBase, logger
Execute using Condor

    This plugin doesn't work with a plain stock-Condor installation, but
    requires a 'qsub' emulation script for Condor, called 'condor_qsub'.
    This script is shipped with the Condor package from NeuroDebian, or can be
    downloaded from its Git repository at

    http://anonscm.debian.org/gitweb/?p=pkg-exppsy/condor.git;a=blob_plain;f=debian/condor_qsub;hb=HEAD

    The plugin_args input to run can be used to control the Condor execution.
    Currently supported options are:

    - template : template to use for batch job submission. This can be an
                 SGE-style script with the (limited) set of options supported
                 by condor_qsub
    - qsub_args : arguments to be prepended to the job execution script in the
                  qsub call
    