def add_edit_managed_option(parser):
    parser.add_argument('--edit-managed', default=False, action='store_true', help='Edit resources marked as managed. Default: False')