import logging
from GameGUI import GameGUI

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Instantiate the GameGUI object, which initializes the game environment and settings
game: GameGUI = GameGUI()
logging.debug("GameGUI object has been instantiated successfully.")

# Enter the main loop of the game, which continues as long as the 'running' attribute of the game object is True
while game.running:
    # Display the current menu, which could be the main menu, options, or any other defined menu in the game's GUI system
    logging.debug("Attempting to display the current menu.")
    game.curr_menu.display_menu()
    logging.debug("Current menu has been displayed successfully.")

    # Execute the main game loop which handles events, updates game state, and renders the game frame by frame
    logging.debug("Entering the game loop.")
    game.game_loop()
    logging.debug("Game loop execution has completed for the current frame.")
