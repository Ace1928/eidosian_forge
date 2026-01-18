from selenium import webdriver

# URL of the web page to capture
url = "https://www.example.com"

# Create a new instance of the Chrome driver
driver = webdriver.Chrome()

# Navigate to the web page
driver.get(url)

# Capture the screenshot
driver.save_screenshot("screenshot.png")

# Close the browser
driver.quit()

print("Screenshot captured successfully.")
