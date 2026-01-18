class CollectionTooLargeException(YaqlException):

    def __init__(self, count):
        super(CollectionTooLargeException, self).__init__('Collection length exceeds {0} elements'.format(count))