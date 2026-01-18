from glance.image_cache import base
class Pruner(base.CacheApp):

    def run(self):
        self.cache.prune()