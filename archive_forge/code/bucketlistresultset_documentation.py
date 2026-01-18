
    A resultset for listing versions within a bucket.  Uses the bucket_lister
    generator function and implements the iterator interface.  This
    transparently handles the results paging from GCS so even if you have
    many thousands of keys within the bucket you can iterate over all
    keys in a reasonably efficient manner.
    