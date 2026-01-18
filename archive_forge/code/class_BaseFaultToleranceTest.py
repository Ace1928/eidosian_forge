import gc
import os
import sys
import threading
import time
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator as thread_coordinator
from tensorflow.python.training import server_lib
class BaseFaultToleranceTest(object):

    def setUp(self, num_workers, num_ps, use_cs=False):
        super(BaseFaultToleranceTest, self).setUp()
        self._cluster = multi_worker_test_base.create_multi_process_cluster(num_workers=num_workers, num_ps=num_ps, rpc_layer='grpc', stream_output=True)
        self._cluster_def = self._cluster.cluster_resolver.cluster_spec().as_dict()
        self._cluster_def['chief'] = ['localhost:%d' % multi_worker_test_base.pick_unused_port()]
        cluster_resolver = SimpleClusterResolver(server_lib.ClusterSpec(self._cluster_def), rpc_layer='grpc')
        if use_cs:
            os.environ['TF_PSS_ENABLE_COORDINATION_SERVICE'] = '1'
        self.strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(cluster_resolver)
        self.cluster_coord = cluster_coordinator.ClusterCoordinator(self.strategy)
        self.thread_coord = thread_coordinator.Coordinator(clean_stop_exception_types=[])
        self.num_workers = num_workers
        self.num_ps = num_ps

    def tearDown(self):
        super(BaseFaultToleranceTest, self).tearDown()
        self._cluster.stop()
        self._cluster = None

    def _restart(self, downtime_secs, job):
        """Kills `job` (index: 0) and restarts it after `downtime_secs`.

    Args:
      downtime_secs: secs before restarting the job.
      job: a string specifying the job to restart.
    """
        self._cluster.kill_task(job, 0)
        time.sleep(downtime_secs)
        self.assertFalse(context.check_alive('/job:%s/replica:0/task:0' % job))
        self._cluster.start_task(job, 0)
        while not context.check_alive('/job:%s/replica:0/task:0' % job):
            time.sleep(1)

    def _restart_in_thread(self, downtime_secs, restart_job):

        def _restart_fn():
            with self.thread_coord.stop_on_exception():
                self._restart(downtime_secs, restart_job)
        restart_thread = threading.Thread(target=_restart_fn)
        restart_thread.start()
        return restart_thread

    def _ensure_threads_closed(self):
        """Ensures worker and preemption threads are closed."""
        running_threads = test_util.get_running_threads()
        self.assertTrue(test_util.has_thread(_WORKER_THREAD_PREFIX, running_threads))
        self.assertIn(_WORKER_PREEMPTION_THREAD_NAME, running_threads)
        if sys.getrefcount(self.cluster_coord) > 2:
            try:
                test_util.show_backref(self.cluster_coord)
            except:
                pass
        self.cluster_coord = None
        self.strategy = None
        gc.collect()
        time.sleep(1)
        running_threads = test_util.get_running_threads()
        self.assertNotIn(_WORKER_PREEMPTION_THREAD_NAME, running_threads)
        self.assertFalse(test_util.has_thread(_WORKER_THREAD_PREFIX, running_threads), 'Worker thread is not stopped properly.')

    def _create_model_and_run_indefinitely(self):
        model = Model(self.cluster_coord)
        model.do_infinite_step.assign(True)
        model.schedule_training_functions(10)
        while self.cluster_coord._cluster.closure_queue._inflight_closure_count < self.num_workers:
            time.sleep(0.1)
        return model

    def testClusterCoordinatorDestroyed(self):
        self._ensure_threads_closed()

    def testWorkerPreemptionBetweenFunctions(self):
        model = Model(self.cluster_coord)
        model.schedule_training_functions(2)
        model.join_training_functions()
        self.assertEqual(model.iterations.numpy(), 2)
        self._restart(downtime_secs=2, job='worker')
        model.schedule_training_functions(2)
        model.join_training_functions()
        self.assertEqual(model.iterations.numpy(), 4)

    def testWorkerPreemptionMidstFunction(self):
        model = Model(self.cluster_coord)
        model.do_infinite_step.assign(True)
        model.schedule_training_functions(4)
        while self.cluster_coord._cluster.closure_queue._inflight_closure_count < self.num_workers:
            time.sleep(0.1)
        self.assertFalse(self.cluster_coord.done())
        self._restart(downtime_secs=2, job='worker')
        model.join_training_functions()
        self.assertGreaterEqual(model.iterations.numpy(), 4)

    def testOneWorkerPreemptionWithCancellation(self):

        @def_function.function
        def normal_function():
            x = random_ops.random_uniform((2, 10))
            y = random_ops.random_uniform((10, 2))
            return math_ops.reduce_mean(math_ops.matmul(x, y))

        @def_function.function
        def error_function():
            x = random_ops.random_uniform((2, 10))
            y = random_ops.random_uniform((10, 2))
            check_ops.assert_non_positive_v2(math_ops.reduce_sum(math_ops.matmul(x, y)))
            return x

        @def_function.function
        def long_function():
            x = random_ops.random_uniform((1000, 1000))
            for _ in math_ops.range(10000):
                a = random_ops.random_uniform((1000, 1000))
                b = random_ops.random_uniform((1000, 1000))
                x += math_ops.matmul(a, b)
            return x
        for _ in range(3):
            self.cluster_coord.schedule(normal_function)
        long_function_result = self.cluster_coord.schedule(long_function)
        self.cluster_coord.schedule(error_function)
        time.sleep(1)
        self._restart(2, 'worker')
        with self.assertRaises(errors.InvalidArgumentError):
            self.cluster_coord.join()
        with self.assertRaises(errors.CancelledError):
            long_function_result.fetch()
        for _ in range(3):
            self.cluster_coord.schedule(normal_function)
        self.cluster_coord.join()
        failure_handler = self.cluster_coord._cluster.failure_handler
        failure_handler.stop()
        failure_handler._preemption_handler_thread.join()

    def testHandleDatasetCreationFailureWithDatasetFn(self):
        model = Model(self.cluster_coord)
        restart_thread = self._restart_in_thread(5, 'worker')
        model.schedule_training_functions(3)
        model.rebuild_iterators()
        model.schedule_training_functions(3)
        model.rebuild_iterators()
        model.schedule_training_functions(3)
        model.join_training_functions()
        self.thread_coord.join([restart_thread])
        self.assertGreaterEqual(model.iterations.numpy(), 3)

    def testHandleDatasetCreationFailureWithDataset(self):
        model = Model(self.cluster_coord)
        restart_thread = self._restart_in_thread(5, 'worker')
        model.schedule_training_functions(3)
        model.rebuild_iterators(use_dataset_fn=False)
        model.schedule_training_functions(3)
        model.rebuild_iterators(use_dataset_fn=False)
        model.schedule_training_functions(3)
        model.join_training_functions()
        self.thread_coord.join([restart_thread])
        self.assertGreaterEqual(model.iterations.numpy(), 3)

    def testWorkerPreemptionErrorType(self):

        @def_function.function
        def worker_train_fn():
            x = random_ops.random_uniform((2, 10))
            y = random_ops.random_uniform((10, 2))
            return math_ops.reduce_mean(math_ops.matmul(x, y))

        def run_fn():
            with self.thread_coord.stop_on_exception():
                with ops.device('/job:worker/replica:0/task:0'):
                    for _ in range(3):
                        for _ in range(3):
                            worker_train_fn()
                        time.sleep(5)
        run_thread = threading.Thread(target=run_fn)
        run_thread.start()
        time.sleep(1)
        self._restart(2, 'worker')
        try:
            self.thread_coord.join([run_thread])
        except (errors.UnavailableError, errors.AbortedError) as e:
            logging.info('Got exception %r, error message is %s', e, e)
            self.assertIn(_RPC_ERROR_FROM_WORKER, str(e))
            self.assertNotIn(_RPC_ERROR_FROM_PS, str(e))
            self.assertTrue('failed to connect to all addresses' in str(e) or 'Unable to find a context_id' in str(e) or 'Socket closed' in str(e) or ('Connection reset by peer' in str(e)) or ('Transport closed' in str(e)))

    def testWorkerPreemptionErrorTypeWithPythonFunction(self):

        def worker_train_fn():
            x = random_ops.random_uniform((2, 10))
            y = random_ops.random_uniform((10, 2))
            return math_ops.reduce_mean(math_ops.matmul(x, y))

        def run_fn():
            with self.thread_coord.stop_on_exception():
                with ops.device('/job:worker/replica:0/task:0'):
                    for _ in range(3):
                        for _ in range(3):
                            worker_train_fn()
                        time.sleep(5)
        run_thread = threading.Thread(target=run_fn)
        run_thread.start()
        time.sleep(1)
        self._restart(2, 'worker')
        try:
            self.thread_coord.join([run_thread])
        except (errors.UnavailableError, errors.AbortedError) as e:
            logging.info('Got exception %r, error message is %s', e, e)
            self.assertIn(_RPC_ERROR_FROM_WORKER, str(e))
            self.assertNotIn(_RPC_ERROR_FROM_PS, str(e))
            self.assertTrue('failed to connect to all addresses' in str(e) or 'Unable to find a context_id' in str(e) or 'Socket closed' in str(e) or ('Connection reset by peer' in str(e)) or ('Transport closed' in str(e)))

    def testPSPreemptionErrorType(self):
        with ops.device('/job:ps/replica:0/task:0'):
            v = variables.Variable(initial_value=random_ops.random_uniform((2, 10)), dtype=dtypes.float32)

        @def_function.function
        def worker_train_fn():
            y = random_ops.random_uniform((10, 2))
            return math_ops.reduce_mean(math_ops.matmul(v, y))

        def run_fn():
            with self.thread_coord.stop_on_exception():
                with ops.device('/job:worker/replica:0/task:0'):
                    for _ in range(3):
                        for _ in range(3):
                            worker_train_fn()
                        time.sleep(5)
        run_thread = threading.Thread(target=run_fn)
        run_thread.start()
        time.sleep(1)
        self._restart(1, 'ps')
        try:
            self.thread_coord.join([run_thread])
        except (errors.UnavailableError, errors.AbortedError) as e:
            logging.info('Got exception %r, error message is %s', e, e)
            self.assertIn(_RPC_ERROR_FROM_PS, str(e))
            if isinstance(e, errors.UnavailableError):
                self.assertTrue('failed to connect to all addresses' in str(e) or 'Socket closed' in str(e) or 'Connection reset by peer' in str(e) or ('Transport closed' in str(e)))
            if isinstance(e, errors.AbortedError):
                self.assertTrue('RecvTensor expects a different device incarnation' in str(e) or 'Unable to find a context_id' in str(e))
            self._ensure_threads_closed()

    def testTwoWorkersPreempted(self):
        if self.num_workers < 2:
            self.skipTest('Worker number is less than 2.')
        model = self._create_model_and_run_indefinitely()
        self.assertFalse(self.cluster_coord.done())
        self._cluster.kill_task('worker', 0)
        self._cluster.kill_task('worker', 1)
        time.sleep(2)
        self.assertFalse(context.check_alive('/job:worker/replica:0/task:0'))
        self.assertFalse(context.check_alive('/job:worker/replica:0/task:1'))
        self._cluster.start_task('worker', 0)
        self._cluster.start_task('worker', 1)
        time.sleep(2)
        self.assertTrue(context.check_alive('/job:worker/replica:0/task:0'))
        self.assertTrue(context.check_alive('/job:worker/replica:0/task:1'))
        model.join_training_functions()
        self.assertGreaterEqual(model.iterations.numpy(), 10)

    def testWorkerContinuousFailure(self):
        model = self._create_model_and_run_indefinitely()
        self.assertFalse(self.cluster_coord.done())
        self._cluster.kill_task('worker', 0)
        time.sleep(2)
        self.assertFalse(context.check_alive('/job:worker/replica:0/task:0'))
        self._cluster.start_task('worker', 0)
        time.sleep(2)
        self.assertTrue(context.check_alive('/job:worker/replica:0/task:0'))
        self._cluster.kill_task('worker', 0)
        time.sleep(2)
        self.assertFalse(context.check_alive('/job:worker/replica:0/task:0'))
        self._cluster.start_task('worker', 0)
        time.sleep(2)
        self.assertTrue(context.check_alive('/job:worker/replica:0/task:0'))
        model.join_training_functions()
        self.assertGreaterEqual(model.iterations.numpy(), 10)

    def testPSFailureWhileRecoveryFromWokerFailure(self):
        model = self._create_model_and_run_indefinitely()
        time.sleep(1)
        self.assertFalse(self.cluster_coord.done())

        def kill(task):
            self._cluster.kill_task(task, 0)
            self.sleep(1)
            self._cluster.start_task(task, 0)
        kill_thread_1 = threading.Thread(target=kill, args=('worker',))
        kill_thread_2 = threading.Thread(target=kill, args=('ps',))
        kill_thread_1.start()
        kill_thread_2.start()
        kill_thread_1.join()
        kill_thread_2.join()
        with self.assertRaises((errors.UnavailableError, errors.InvalidArgumentError)):
            model.join_training_functions()

    def testNumpyFetchedAfterWorkerFailure(self):
        with self.strategy.scope():
            v = variables.Variable(initial_value=0, dtype=dtypes.int32)

        @def_function.function
        def worker_fn():
            return (v + 1, v - 1)
        remote_value = self.cluster_coord.schedule(worker_fn)
        self.assertEqual((1, -1), remote_value.fetch())
        self._cluster.kill_task('worker', 0)
        self.assertEqual((1, -1), remote_value.fetch())

    def testTensorGotAfterWorkerFailure(self):
        with self.strategy.scope():
            v = variables.Variable(initial_value=0, dtype=dtypes.int32)

        @def_function.function
        def worker_fn():
            return (v + 1, v - 1)
        remote_value = self.cluster_coord.schedule(worker_fn)
        fetched = remote_value.get()[0]
        self.assertIsInstance(fetched, tensor.Tensor)
        self.assertEqual(fetched.device, '/job:chief/replica:0/task:0/device:CPU:0')
        self.assertEqual((1, -1), remote_value.get())
        remote_value.get()[0].numpy()
        values = remote_value._values[0]
        self.assertIsInstance(values, tensor.Tensor)
        self.assertRegex(values.device, '/job:worker/replica:0/task:[0-1]/device:CPU:0')
        self.assertEqual((1, -1), remote_value._values)
        remote_value._values[0].numpy()
        for i in range(self.num_workers):
            self._cluster.kill_task('worker', i)
        time.sleep(5)
        remote_value.get()[0].numpy()
        self.assertEqual((1, -1), remote_value.get())
        with self.assertRaises(errors.UnavailableError) as cm:
            remote_value._values[0].numpy()
        self.assertIn('failed to connect to all addresses', cm.exception.message)
        self.assertIn('/job:worker/replica:0/task:', cm.exception.message)

    def testFetchFromPSAfterWorkerFailure(self):
        model = Model(self.cluster_coord)

        def kill_after_delay():
            time.sleep(3)
            logging.info('Killing worker 0')
            self._cluster.kill_task('worker', 0)
            time.sleep(1)
            logging.info('Restarting worker 0')
            self._cluster.start_task('worker', 0)
        kill_thread = threading.Thread(target=kill_after_delay)
        kill_thread.start()
        model.do_infinite_step.assign(True)
        model.schedule_training_functions(1)
        num_reads = 0
        num_reads_after_restart = 0
        read_interval_secs = 0.1
        worker_has_stopped = False
        while num_reads_after_restart <= 5 and num_reads < 200:
            worker_up = context.check_alive('/job:worker/replica:0/task:0')
            if not worker_up:
                worker_has_stopped = True
            if worker_up and worker_has_stopped:
                num_reads_after_restart += 1
            model.join_training_functions()
            start = time.time()
            while time.time() < start + read_interval_secs:
                model.iterations.read_value()
            num_reads += 1
            model.do_infinite_step.assign(True)
            model.schedule_training_functions(1)

    def testClusterStateNotDisrupted(self):
        self.thread_coord = thread_coordinator.Coordinator(clean_stop_exception_types=[])
        self.testWorkerPreemptionMidstFunction()
        self.thread_coord = thread_coordinator.Coordinator(clean_stop_exception_types=[])
        self.testWorkerPreemptionErrorType()

    def _run_and_kill_ps_task(self):
        self._create_model_and_run_indefinitely()
        self._cluster.kill_task('ps', 0)
        while self.cluster_coord._cluster.closure_queue._error is None:
            time.sleep(1)
        logging.info('Trying to join, expecting error')

    def testJoinRaisesUnavailableErrorAtPsFailure(self):
        self._run_and_kill_ps_task()
        with self.assertRaises((errors.UnavailableError, errors.NotFoundError, errors.FailedPreconditionError)):
            self.cluster_coord.join()

    def testScheduleRaisesUnavailableErrorAtPsFailure(self):
        self._run_and_kill_ps_task()
        with self.assertRaises((errors.UnavailableError, errors.NotFoundError, errors.FailedPreconditionError)):
            self.cluster_coord.schedule(def_function.function(lambda: None))

    def testWorkerExecutionAfterPsFailureRaisesExpectedError(self):
        model = self._create_model_and_run_indefinitely()
        for i in range(self.num_ps):
            self._cluster.kill_task('ps', i)
        while self.cluster_coord._cluster.closure_queue._error is None:
            time.sleep(1)

        @def_function.function
        def trivial_function():
            return model.iterations + 1
        for i in range(self.num_workers):
            try:
                with ops.device('/job:worker/replica:0/task:{}'.format(i)):
                    trivial_function()
            except Exception as e:
                if cluster_coordinator._is_ps_failure(e):
                    if i < self.num_workers - 1:
                        continue
                    return
            raise AssertionError('Executing a function after PS fails, should result in a PS failure.')

    def testAsyncWaitIsNoOp(self):
        if self.num_workers < 2:
            self.skipTest('Worker number is less than 2.')
        model = self._create_model_and_run_indefinitely()
        self.assertFalse(self.cluster_coord.done())
        self._cluster.kill_task('worker', 0)
        time.sleep(2)
        self.assertFalse(context.check_alive('/job:worker/replica:0/task:0'))
        context.async_wait()
        model.join_training_functions()
        self.assertGreaterEqual(model.iterations.numpy(), 10)
        self._cluster.start_task('worker', 0)