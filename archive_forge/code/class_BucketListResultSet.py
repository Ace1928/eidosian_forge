from boto.compat import unquote_str
class BucketListResultSet(object):
    """
    A resultset for listing keys within a bucket.  Uses the bucket_lister
    generator function and implements the iterator interface.  This
    transparently handles the results paging from S3 so even if you have
    many thousands of keys within the bucket you can iterate over all
    keys in a reasonably efficient manner.
    """

    def __init__(self, bucket=None, prefix='', delimiter='', marker='', headers=None, encoding_type=None):
        self.bucket = bucket
        self.prefix = prefix
        self.delimiter = delimiter
        self.marker = marker
        self.headers = headers
        self.encoding_type = encoding_type

    def __iter__(self):
        return bucket_lister(self.bucket, prefix=self.prefix, delimiter=self.delimiter, marker=self.marker, headers=self.headers, encoding_type=self.encoding_type)