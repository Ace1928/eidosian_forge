from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import errno
import os
import pdb
import sys
import textwrap
import traceback
from absl import command_name
from absl import flags
from absl import logging
def _run_main(main, argv):
    """Calls main, optionally with pdb or profiler."""
    if FLAGS.run_with_pdb:
        sys.exit(pdb.runcall(main, argv))
    elif FLAGS.run_with_profiling or FLAGS.profile_file:
        import atexit
        if FLAGS.use_cprofile_for_profiling:
            import cProfile as profile
        else:
            import profile
        profiler = profile.Profile()
        if FLAGS.profile_file:
            atexit.register(profiler.dump_stats, FLAGS.profile_file)
        else:
            atexit.register(profiler.print_stats)
        retval = profiler.runcall(main, argv)
        sys.exit(retval)
    else:
        sys.exit(main(argv))