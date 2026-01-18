import functools
import random
import time
class CloudRetry:
    """
    The base class to be used by other cloud providers to provide a backoff/retry decorator based on status codes.
    """
    base_class = type(None)

    @staticmethod
    def status_code_from_exception(error):
        """
        Returns the Error 'code' from an exception.
        Args:
          error: The Exception from which the error code is to be extracted.
            error will be an instance of class.base_class.
        """
        raise NotImplementedError()

    @staticmethod
    def found(response_code, catch_extra_error_codes=None):

        def _is_iterable():
            try:
                iter(catch_extra_error_codes)
            except TypeError:
                return False
            else:
                return True
        return _is_iterable() and response_code in catch_extra_error_codes

    @classmethod
    def base_decorator(cls, retries, found, status_code_from_exception, catch_extra_error_codes, sleep_time_generator):

        def retry_decorator(func):

            @functools.wraps(func)
            def _retry_wrapper(*args, **kwargs):
                partial_func = functools.partial(func, *args, **kwargs)
                return _retry_func(func=partial_func, sleep_time_generator=sleep_time_generator, retries=retries, catch_extra_error_codes=catch_extra_error_codes, found_f=found, status_code_from_except_f=status_code_from_exception, base_class=cls.base_class)
            return _retry_wrapper
        return retry_decorator

    @classmethod
    def exponential_backoff(cls, retries=10, delay=3, backoff=2, max_delay=60, catch_extra_error_codes=None):
        """Wrap a callable with retry behavior.
        Args:
            retries (int): Number of times to retry a failed request before giving up
                default=10
            delay (int or float): Initial delay between retries in seconds
                default=3
            backoff (int or float): backoff multiplier e.g. value of 2 will  double the delay each retry
                default=2
            max_delay (int or None): maximum amount of time to wait between retries.
                default=60
            catch_extra_error_codes: Additional error messages to catch, in addition to those which may be defined by a subclass of CloudRetry
                default=None
        Returns:
            Callable: A generator that calls the decorated function using an exponential backoff.
        """
        sleep_time_generator = BackoffIterator(delay=delay, backoff=backoff, max_delay=max_delay)
        return cls.base_decorator(retries=retries, found=cls.found, status_code_from_exception=cls.status_code_from_exception, catch_extra_error_codes=catch_extra_error_codes, sleep_time_generator=sleep_time_generator)

    @classmethod
    def jittered_backoff(cls, retries=10, delay=3, backoff=2.0, max_delay=60, catch_extra_error_codes=None):
        """Wrap a callable with retry behavior.
        Args:
            retries (int): Number of times to retry a failed request before giving up
                default=10
            delay (int or float): Initial delay between retries in seconds
                default=3
            backoff (int or float): backoff multiplier e.g. value of 2 will  double the delay each retry
                default=2.0
            max_delay (int or None): maximum amount of time to wait between retries.
                default=60
            catch_extra_error_codes: Additional error messages to catch, in addition to those which may be defined by a subclass of CloudRetry
                default=None
        Returns:
            Callable: A generator that calls the decorated function using using a jittered backoff strategy.
        """
        sleep_time_generator = BackoffIterator(delay=delay, backoff=backoff, max_delay=max_delay, jitter=True)
        return cls.base_decorator(retries=retries, found=cls.found, status_code_from_exception=cls.status_code_from_exception, catch_extra_error_codes=catch_extra_error_codes, sleep_time_generator=sleep_time_generator)