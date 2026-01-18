import logging
class IntegerSet(object):

    @staticmethod
    def get_interval():
        return (None, None, 1)

    @staticmethod
    def is_continuous():
        return False

    @staticmethod
    def is_integer():
        return True

    @staticmethod
    def is_binary():
        return False