def add_all_projects_option(parser):
    parser.add_argument('--all-projects', default=False, action='store_true', help='Show results from all projects. Default: False')