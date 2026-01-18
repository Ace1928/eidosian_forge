import logging
class RealSet(object):

    @staticmethod
    def get_interval():
        return (None, None, 0)

    @staticmethod
    def is_continuous():
        return True

    @staticmethod
    def is_integer():
        return False

    @staticmethod
    def is_binary():
        return False