class VersionedBucketListResultSet(object):
    """
    A resultset for listing versions within a bucket.  Uses the bucket_lister
    generator function and implements the iterator interface.  This
    transparently handles the results paging from GCS so even if you have
    many thousands of keys within the bucket you can iterate over all
    keys in a reasonably efficient manner.
    """

    def __init__(self, bucket=None, prefix='', delimiter='', marker='', generation_marker='', headers=None):
        self.bucket = bucket
        self.prefix = prefix
        self.delimiter = delimiter
        self.marker = marker
        self.generation_marker = generation_marker
        self.headers = headers

    def __iter__(self):
        return versioned_bucket_lister(self.bucket, prefix=self.prefix, delimiter=self.delimiter, marker=self.marker, generation_marker=self.generation_marker, headers=self.headers)