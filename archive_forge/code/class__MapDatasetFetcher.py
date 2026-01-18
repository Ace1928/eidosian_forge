class _MapDatasetFetcher(_BaseDatasetFetcher):

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, '__getitems__') and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)