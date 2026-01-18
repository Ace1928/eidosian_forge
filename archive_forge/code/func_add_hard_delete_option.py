def add_hard_delete_option(parser):
    parser.add_argument('--hard-delete', default=False, action='store_true', help='Delete zone along-with backend zone resources (i.e. files). Default: False')