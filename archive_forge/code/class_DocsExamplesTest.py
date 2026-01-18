import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
class DocsExamplesTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mpl_blocker = MatplotlibBlocker()

    def setUp(self):
        sys.meta_path.insert(0, self._mpl_blocker)

    def tearDown(self):
        sys.meta_path.remove(self._mpl_blocker)

    def test_mandelbrot(self):
        with captured_stdout():
            from timeit import default_timer as timer
            try:
                from matplotlib.pylab import imshow, show
                have_mpl = True
            except ImportError:
                have_mpl = False
            import numpy as np
            from numba import jit

            @jit(nopython=True)
            def mandel(x, y, max_iters):
                """
                Given the real and imaginary parts of a complex number,
                determine if it is a candidate for membership in the Mandelbrot
                set given a fixed number of iterations.
                """
                i = 0
                c = complex(x, y)
                z = 0j
                for i in range(max_iters):
                    z = z * z + c
                    if z.real * z.real + z.imag * z.imag >= 4:
                        return i
                return 255

            @jit(nopython=True)
            def create_fractal(min_x, max_x, min_y, max_y, image, iters):
                height = image.shape[0]
                width = image.shape[1]
                pixel_size_x = (max_x - min_x) / width
                pixel_size_y = (max_y - min_y) / height
                for x in range(width):
                    real = min_x + x * pixel_size_x
                    for y in range(height):
                        imag = min_y + y * pixel_size_y
                        color = mandel(real, imag, iters)
                        image[y, x] = color
                return image
            image = np.zeros((500 * 2, 750 * 2), dtype=np.uint8)
            s = timer()
            create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
            e = timer()
            print(e - s)
            if have_mpl:
                imshow(image)
                show()

    def test_moving_average(self):
        with captured_stdout():
            import numpy as np
            from numba import guvectorize

            @guvectorize(['void(float64[:], intp[:], float64[:])'], '(n),()->(n)')
            def move_mean(a, window_arr, out):
                window_width = window_arr[0]
                asum = 0.0
                count = 0
                for i in range(window_width):
                    asum += a[i]
                    count += 1
                    out[i] = asum / count
                for i in range(window_width, len(a)):
                    asum += a[i] - a[i - window_width]
                    out[i] = asum / count
            arr = np.arange(20, dtype=np.float64).reshape(2, 10)
            print(arr)
            print(move_mean(arr, 3))

    def test_nogil(self):
        with captured_stdout():
            import math
            import threading
            from timeit import repeat
            import numpy as np
            from numba import jit
            nthreads = 4
            size = 10 ** 6

            def func_np(a, b):
                """
                Control function using Numpy.
                """
                return np.exp(2.1 * a + 3.2 * b)

            @jit('void(double[:], double[:], double[:])', nopython=True, nogil=True)
            def inner_func_nb(result, a, b):
                """
                Function under test.
                """
                for i in range(len(result)):
                    result[i] = math.exp(2.1 * a[i] + 3.2 * b[i])

            def timefunc(correct, s, func, *args, **kwargs):
                """
                Benchmark *func* and print out its runtime.
                """
                print(s.ljust(20), end=' ')
                res = func(*args, **kwargs)
                if correct is not None:
                    assert np.allclose(res, correct), (res, correct)
                print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs), number=5, repeat=2)) * 1000))
                return res

            def make_singlethread(inner_func):
                """
                Run the given function inside a single thread.
                """

                def func(*args):
                    length = len(args[0])
                    result = np.empty(length, dtype=np.float64)
                    inner_func(result, *args)
                    return result
                return func

            def make_multithread(inner_func, numthreads):
                """
                Run the given function inside *numthreads* threads, splitting
                its arguments into equal-sized chunks.
                """

                def func_mt(*args):
                    length = len(args[0])
                    result = np.empty(length, dtype=np.float64)
                    args = (result,) + args
                    chunklen = (length + numthreads - 1) // numthreads
                    chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in args] for i in range(numthreads)]
                    threads = [threading.Thread(target=inner_func, args=chunk) for chunk in chunks]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    return result
                return func_mt
            func_nb = make_singlethread(inner_func_nb)
            func_nb_mt = make_multithread(inner_func_nb, nthreads)
            a = np.random.rand(size)
            b = np.random.rand(size)
            correct = timefunc(None, 'numpy (1 thread)', func_np, a, b)
            timefunc(correct, 'numba (1 thread)', func_nb, a, b)
            timefunc(correct, 'numba (%d threads)' % nthreads, func_nb_mt, a, b)

    def test_vectorize_one_signature(self):
        with captured_stdout():
            from numba import vectorize, float64

            @vectorize([float64(float64, float64)])
            def f(x, y):
                return x + y

    def test_vectorize_multiple_signatures(self):
        with captured_stdout():
            from numba import vectorize, int32, int64, float32, float64
            import numpy as np

            @vectorize([int32(int32, int32), int64(int64, int64), float32(float32, float32), float64(float64, float64)])
            def f(x, y):
                return x + y
            a = np.arange(6)
            result = f(a, a)
            self.assertIsInstance(result, np.ndarray)
            correct = np.array([0, 2, 4, 6, 8, 10])
            np.testing.assert_array_equal(result, correct)
            a = np.linspace(0, 1, 6)
            result = f(a, a)
            self.assertIsInstance(result, np.ndarray)
            correct = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
            np.testing.assert_allclose(result, correct)
            a = np.arange(12).reshape(3, 4)
            result1 = f.reduce(a, axis=0)
            result2 = f.reduce(a, axis=1)
            result3 = f.accumulate(a)
            result4 = f.accumulate(a, axis=1)
            self.assertIsInstance(result1, np.ndarray)
            correct = np.array([12, 15, 18, 21])
            np.testing.assert_array_equal(result1, correct)
            self.assertIsInstance(result2, np.ndarray)
            correct = np.array([6, 22, 38])
            np.testing.assert_array_equal(result2, correct)
            self.assertIsInstance(result3, np.ndarray)
            correct = np.array([[0, 1, 2, 3], [4, 6, 8, 10], [12, 15, 18, 21]])
            np.testing.assert_array_equal(result3, correct)
            self.assertIsInstance(result4, np.ndarray)
            correct = np.array([[0, 1, 3, 6], [4, 9, 15, 22], [8, 17, 27, 38]])
            np.testing.assert_array_equal(result4, correct)

    def test_guvectorize(self):
        with captured_stdout():
            from numba import guvectorize, int64
            import numpy as np

            @guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
            def g(x, y, res):
                for i in range(x.shape[0]):
                    res[i] = x[i] + y
            a = np.arange(5)
            result = g(a, 2)
            self.assertIsInstance(result, np.ndarray)
            correct = np.array([2, 3, 4, 5, 6])
            np.testing.assert_array_equal(result, correct)
            a = np.arange(6).reshape(2, 3)
            result1 = g(a, 10)
            result2 = g(a, np.array([10, 20]))
            g(a, np.array([10, 20]))
            self.assertIsInstance(result1, np.ndarray)
            correct = np.array([[10, 11, 12], [13, 14, 15]])
            np.testing.assert_array_equal(result1, correct)
            self.assertIsInstance(result2, np.ndarray)
            correct = np.array([[10, 11, 12], [23, 24, 25]])
            np.testing.assert_array_equal(result2, correct)

    def test_guvectorize_scalar_return(self):
        with captured_stdout():
            from numba import guvectorize, int64
            import numpy as np

            @guvectorize([(int64[:], int64, int64[:])], '(n),()->()')
            def g(x, y, res):
                acc = 0
                for i in range(x.shape[0]):
                    acc += x[i] + y
                res[0] = acc
            a = np.arange(5)
            result = g(a, 2)
            self.assertIsInstance(result, np.integer)
            self.assertEqual(result, 20)

    def test_guvectorize_overwrite(self):
        with captured_stdout():
            from numba import guvectorize, float64
            import numpy as np

            @guvectorize([(float64[:], float64[:])], '()->()')
            def init_values(invals, outvals):
                invals[0] = 6.5
                outvals[0] = 4.2
            invals = np.zeros(shape=(3, 3), dtype=np.float64)
            outvals = init_values(invals)
            self.assertIsInstance(invals, np.ndarray)
            correct = np.array([[6.5, 6.5, 6.5], [6.5, 6.5, 6.5], [6.5, 6.5, 6.5]])
            np.testing.assert_array_equal(invals, correct)
            self.assertIsInstance(outvals, np.ndarray)
            correct = np.array([[4.2, 4.2, 4.2], [4.2, 4.2, 4.2], [4.2, 4.2, 4.2]])
            np.testing.assert_array_equal(outvals, correct)
            invals = np.zeros(shape=(3, 3), dtype=np.float32)
            outvals = init_values(invals)
            print(invals)
            self.assertIsInstance(invals, np.ndarray)
            correct = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
            np.testing.assert_array_equal(invals, correct)
            self.assertIsInstance(outvals, np.ndarray)
            correct = np.array([[4.2, 4.2, 4.2], [4.2, 4.2, 4.2], [4.2, 4.2, 4.2]])
            np.testing.assert_array_equal(outvals, correct)

            @guvectorize([(float64[:], float64[:])], '()->()', writable_args=('invals',))
            def init_values(invals, outvals):
                invals[0] = 6.5
                outvals[0] = 4.2
            invals = np.zeros(shape=(3, 3), dtype=np.float32)
            outvals = init_values(invals)
            print(invals)
            self.assertIsInstance(invals, np.ndarray)
            correct = np.array([[6.5, 6.5, 6.5], [6.5, 6.5, 6.5], [6.5, 6.5, 6.5]])
            np.testing.assert_array_equal(invals, correct)
            self.assertIsInstance(outvals, np.ndarray)
            correct = np.array([[4.2, 4.2, 4.2], [4.2, 4.2, 4.2], [4.2, 4.2, 4.2]])
            np.testing.assert_array_equal(outvals, correct)

    def test_vectorize_dynamic(self):
        with captured_stdout():
            from numba import vectorize

            @vectorize
            def f(x, y):
                return x * y
            result = f(3, 4)
            print(f.types)
            self.assertEqual(result, 12)
            if IS_WIN32:
                correct = ['ll->q']
            else:
                correct = ['ll->l']
            self.assertEqual(f.types, correct)
            result = f(1.0, 2.0)
            print(f.types)
            self.assertEqual(result, 2.0)
            if IS_WIN32:
                correct = ['ll->q', 'dd->d']
            else:
                correct = ['ll->l', 'dd->d']
            self.assertEqual(f.types, correct)
            result = f(1, 2.0)
            print(f.types)
            self.assertEqual(result, 2.0)
            if IS_WIN32:
                correct = ['ll->q', 'dd->d']
            else:
                correct = ['ll->l', 'dd->d']
            self.assertEqual(f.types, correct)

            @vectorize
            def g(a, b):
                return a / b
            print(g(2.0, 3.0))
            print(g(2, 3))
            print(g.types)
            correct = ['dd->d']
            self.assertEqual(g.types, correct)

    def test_guvectorize_dynamic(self):
        with captured_stdout():
            from numba import guvectorize
            import numpy as np

            @guvectorize('(n),()->(n)')
            def g(x, y, res):
                for i in range(x.shape[0]):
                    res[i] = x[i] + y
            x = np.arange(5, dtype=np.int64)
            y = 10
            res = np.zeros_like(x)
            g(x, y, res)
            print(g.types)
            correct = np.array([10, 11, 12, 13, 14])
            np.testing.assert_array_equal(res, correct)
            if IS_WIN32:
                correct = ['qq->q']
            else:
                correct = ['ll->l']
            self.assertEqual(g.types, correct)
            x = np.arange(5, dtype=np.double)
            y = 2.2
            res = np.zeros_like(x)
            g(x, y, res)
            print(g.types)
            if IS_WIN32:
                correct = ['qq->q', 'dd->d']
            else:
                correct = ['ll->l', 'dd->d']
            self.assertEqual(g.types, correct)
            x = np.arange(5, dtype=np.int64)
            y = 2.2
            res = np.zeros_like(x)
            g(x, y, res)
            print(res)
            correct = np.array([2, 3, 4, 5, 6])
            np.testing.assert_array_equal(res, correct)