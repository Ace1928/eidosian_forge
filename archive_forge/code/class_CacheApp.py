from glance.image_cache import ImageCache
class CacheApp(object):

    def __init__(self):
        self.cache = ImageCache()