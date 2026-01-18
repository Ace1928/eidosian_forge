# screen dimensions
CELL_SIZE: int = (
    40  # This integer represents the size of each cell in the game grid, measured in pixels.
)
NO_OF_CELLS: int = (
    20  # This integer denotes the number of cells along one dimension of the square game grid.
)
BANNER_HEIGHT: int = (
    2  # This integer specifies the height of the banner area at the top of the game screen, measured in cells.
)

# button + text field dimensions
BTN_WIDTH: int = (
    200  # This integer represents the width of buttons in the game interface, measured in pixels.
)
BTN_HEIGHT: int = (
    80  # This integer represents the height of buttons in the game interface, measured in pixels.
)
TXT_WIDTH: int = (
    150  # This integer represents the width of text fields in the game interface, measured in pixels.
)
TXT_HEIGHT: int = (
    30  # This integer represents the height of text fields in the game interface, measured in pixels.
)

# seed
USER_SEED: int = (
    76767  # This integer is used as a seed for random number generation to ensure reproducibility.
)

# COLORS
# 1) buttons
BTN_COLOR: Tuple[int, int, int] = (
    118,
    131,
    163,
)  # This tuple of three integers represents the RGB color of buttons when they are in their default state.
BTN_HOVER: Tuple[int, int, int] = (
    94,
    219,
    111,
)  # This tuple of three integers represents the RGB color of buttons when the mouse cursor hovers over them.
BTN_CLICKED: Tuple[int, int, int] = (
    94,
    219,
    111,
)  # This tuple of three integers represents the RGB color of buttons when they are clicked.

# 2) text field
TXT_PASSIVE: Tuple[int, int, int] = (
    162,
    163,
    163,
)  # This tuple of three integers represents the RGB color of text fields when they are not active.
TXT_ACTIVE: Tuple[int, int, int] = (
    94,
    219,
    111,
)  # This tuple of three integers represents the RGB color of text fields when they are active.

# 3) snake + path + fruit
SNAKE_COLOR: Tuple[int, int, int] = (
    235,
    235,
    235,
)  # This tuple of three integers represents the RGB color of the snake's body.
SNAKE_HEAD_COLOR: Tuple[int, int, int] = (
    106,
    164,
    189,
)  # This tuple of three integers represents the RGB color of the snake's head.
SNAKE_COLOR = (235, 235, 235)
SNAKE_HEAD_COLOR = (106, 164, 189)
FRUIT_COLOR = (219, 90, 101)
PATHCOLOR = (41, 255, 3, 50)

# 4) related to screen
WINDOW_COLOR = (40, 40, 41)
WHITE = (241, 241, 241)
MENU_COLOR = (94, 219, 111)
BANNER_COLOR = (189, 189, 189)
TITLE_COLOR = (133, 209, 242)
