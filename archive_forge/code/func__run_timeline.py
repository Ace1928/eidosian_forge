import logging
import os
import time
from ray.util.debug import log_once
from ray.rllib.utils.framework import try_import_tf
def _run_timeline(sess, ops, debug_name, feed_dict=None, timeline_dir=None):
    if feed_dict is None:
        feed_dict = {}
    if timeline_dir:
        from tensorflow.python.client import timeline
        try:
            run_options = tf1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        except AttributeError:
            run_options = None
            if log_once('tf1.RunOptions_not_available'):
                logger.exception('Can not access tf.RunOptions.FULL_TRACE. This may be because you have used `ray.init(local_mode=True)`. RLlib will use timeline without `options=tf.RunOptions.FULL_TRACE`.')
        run_metadata = tf1.RunMetadata()
        start = time.time()
        fetches = sess.run(ops, options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        global _count
        outf = os.path.join(timeline_dir, 'timeline-{}-{}-{}.json'.format(debug_name, os.getpid(), _count % 10))
        _count += 1
        trace_file = open(outf, 'w')
        logger.info('Wrote tf timeline ({} s) to {}'.format(time.time() - start, os.path.abspath(outf)))
        trace_file.write(trace.generate_chrome_trace_format())
    else:
        if log_once('tf_timeline'):
            logger.info('Executing TF run without tracing. To dump TF timeline traces to disk, set the TF_TIMELINE_DIR environment variable.')
        fetches = sess.run(ops, feed_dict=feed_dict)
    return fetches