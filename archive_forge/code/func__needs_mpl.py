def _needs_mpl(func):

    def wrapper():
        if not _has_mpl:
            raise ImportError('The drawer style module requires matplotlib. You can install matplotlib via \n\n   pip install matplotlib')
        func()
    return wrapper